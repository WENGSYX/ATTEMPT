import torch
import temp.sampler as sampler
from torch.utils.data import dataloader
import transformers
from transformers import BertTokenizer, ElectraTokenizer, AlbertTokenizer, RobertaTokenizer, XLMTokenizer, \
    XLNetTokenizer
from tqdm import tqdm
from temp.base import BaseTrainer
import codecs
from temp import TEMPBert, TEMPElectra, TEMPAlbert, TEMPRoberta, TEMPXLNet, TEMPXLM
import json
import temp
import scorer
import os
import time
import torch.nn.functional as F
class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
"""
def nxloss(pos,neg,temperature=0.5):
    normalized_rep1 = F.normalize(pos)
    normalized_rep2 = F.normalize(neg)
    dis_matrix = torch.bmm(normalized_rep1, torch.transpose(normalized_rep2, 1, 2)) / temperature
    pos = [torch.diag(i) for i in dis_matrix]
    pos = torch.stack(pos, dim=0)
    dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
    loss = (torch.log(dedominator) - pos).mean()
    return loss
"""

def nxloss(pos,neg,temperature=0.5):
    normalized_rep1 = F.normalize(pos)
    normalized_rep2 = F.normalize(neg)
    dis_matrix = torch.bmm(normalized_rep1, torch.transpose(normalized_rep2, 1, 2)) / temperature

    dedominator = torch.sum(torch.exp(dis_matrix), dim=1)
    loss = (torch.log(dedominator)).mean()
    return loss

def log(text,path):
    with open(path+'/log.txt','a',encoding='utf-8') as f:
        f.write('-----------------{}-----------------'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write('\n')
        for i in text:
            f.write(i)
            print(i)
            f.write('\n')
        f.write('\n')
import os
def log_start(log_name):
    if log_name == '':
        log_name = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    else:
        try:
            os.mkdir('./log/' + log_name)
        except:
            log_name += time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            os.mkdir('./log/' + log_name)
    os.makedirs('./log/' + log_name+'/python')
    for root, dirs, files in os.walk('./'):
        if 'log' not in root:
            for file in files:
                if os.path.splitext(file)[1] == '.py':
                    with open('./log/' + log_name + '/python/'+file, 'a', encoding='utf-8') as f:
                        with open(root+'/'+file,'r',encoding='utf-8') as f2:
                            file = f2.read()
                        f.write(file)

    path = './log/' + log_name
    with open(path+'/log.txt', 'a', encoding='utf-8') as f:
        f.write(log_name)
        f.write('\n')
    return path
class Trainer(BaseTrainer):
    models = {'bert': TEMPBert, 'electra': TEMPElectra, 'albert': TEMPAlbert, 'roberta': TEMPRoberta,
              'xlnet': TEMPXLNet, 'xlm': TEMPXLM}
    tokenizers = {'bert': BertTokenizer, 'electra': ElectraTokenizer, 'albert': AlbertTokenizer,
                  'roberta': RobertaTokenizer, 'xlnet': XLNetTokenizer, 'xlm': XLMTokenizer}

    def __init__(self, args,log_name):
        super().__init__(args)
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            # TAXONOMY FILE FORMAT: hypernym <TAB> term
            tax_lines = f.readlines()
        tax_pairs = [line.strip().split("\t") for line in tax_lines]

        self.tax_graph = sampler.TaxStruct(tax_pairs)
        self.sampler = sampler.Sampler(self.tax_graph)

        self.model = self.models[args.model_type].from_pretrained(self.args.pretrained_path,
                                                                  # gradient_checkpointing=True,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False
                                                                  # force_download=True
                                                                  )
        self._tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_path)
        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            self._word2des = json.load(fp)
        self.path=log_start(log_name)
    def enc_dec_get_params_for_prompt_optimization(self,module: torch.nn.Module):
        params = []
        for t in module.named_modules():
            if "prompt" in t[0]:
                params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})

        for t in module.named_parameters():
            if "prompt" not in t[0]:
                t[1].requires_grad_(False)

        return params
    def train(self,eval_args,split_args):
        e = temp.Eval(eval_args)
        print('start')

        #param_groups = self.enc_dec_get_params_for_prompt_optimization(self.model)
        optimizer = transformers.AdamW(self.model.parameters(),
                                       lr=self.args.lr,  # args.learning_rate - default is 5e-5
                                       eps=self.args.eps  # args.adam_epsilon  - default is 1e-8
                                       )
        dataset = sampler.Dataset(self.sampler,
                                  tokenizer=self._tokenizer,
                                  word2des=self._word2des,
                                  padding_max=self.args.padding_max,
                                  margin_beta=self.args.margin_beta)
        data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=len(data_loader) * self.args.epochs)
        self.model.cuda()
        loss_count = 0
        loss_max = 0
        losses = AverageMeter()
        tk = tqdm(range(self.args.epochs), desc="Training", total=self.args.epochs)
        for epoch in tk:
            dataset = sampler.Dataset(self.sampler,
                                      tokenizer=self._tokenizer,
                                      word2des=self._word2des,
                                      padding_max=self.args.padding_max,
                                      margin_beta=self.args.margin_beta)
            data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            loss_all = 0.0

            for step,batch in enumerate(data_loader):
                optimizer.zero_grad()
                pos_output,pos_bert = self.model(input_ids=batch["pos_ids"].cuda(), token_type_ids=batch["pos_type_ids"].cuda(),
                                        attention_mask=batch["pos_attn_masks"].cuda())
                neg_output,neg_bert = self.model(input_ids=batch["neg_ids"].cuda(), token_type_ids=batch["neg_type_ids"].cuda(),
                                        attention_mask=batch["neg_attn_masks"].cuda())


                loss = self.model.margin_loss_fct(pos_output, neg_output, batch["margin"].cuda())
                #loss2 = -torch.log((nxloss(neg_bert.last_hidden_state,neg_bert2.last_hidden_state).clamp(min=1e-7)/((nxloss(pos_bert.last_hidden_state,neg_bert.last_hidden_state)+nxloss(pos_bert.last_hidden_state,neg_bert2.last_hidden_state))/2).clamp(min=1e-7)).clamp(min=1e-7))
                #loss += loss2
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_all += loss.item()
                self._log_tensorboard(self.args.log_label, "", loss.item(), loss_count)
                losses.update(loss.item())
                tk.set_postfix(loss=losses.avg,now_loss=loss.item())
                loss_count += 1
            loss_max = max(loss_max, loss_all)
            if epoch % 10 == 0 and epoch != 0:
                self.model.eval()
                e.model = self.model
                results = e.predict()
                with open(split_args.eval_path, encoding='utf-8') as fp:
                    lines = fp.readlines()
                trues = [line.strip() for line in lines]
                acc, mrr, wu_p, wrong_set = scorer.score(results, self.tax_graph, trues)
                log(["epoch: {}  acc: {}  mrr: {}  wu_p: {} loss_avg: {}".format(epoch, acc, mrr, wu_p,losses.avg)],self.path)
                path = '{}/{}_{}_{}_{}_{}'.format(self.path,self.args.log_label, str(epoch), str(acc), str(mrr), str(wu_p))
                os.mkdir(path)
                self.save_model(path)
                self.model.train()

        return True

    def save_model(self,path):
        self.model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)
