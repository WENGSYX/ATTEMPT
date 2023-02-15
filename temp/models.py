from transformers.models.bert import BertTokenizer, BertConfig, BertPreTrainedModel,BertLayer
from transformers.models.electra import ElectraPreTrainedModel, ElectraModel, ElectraConfig
from transformers.models.albert import AlbertModel, AlbertPreTrainedModel, AlbertConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel, RobertaConfig
from transformers.models.xlnet import XLNetModel, XLNetPreTrainedModel, XLNetConfig, XLNetForSequenceClassification
from transformers.models.xlm import XLMModel, XLMPreTrainedModel, XLMConfig, XLMForSequenceClassification
from transformers.modeling_utils import SequenceSummary
# from transformers.models.xlm_roberta
# from transformers.models.bart import BartTokenizer, BartPretrainedModel, BartModel, BartConfig
from torch import nn
import torch
from transformers import AutoModel,AutoTokenizer
from transformers.activations import get_activation
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
BERT_START_DOCSTRING = r"""
    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)
    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.
    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.prompt_embeddings = nn.Embedding(100, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def get_input_embeds(self, input_ids):

        p_embeds = None

        prompt_mask = (input_ids < 0).long()
        prompt_ids = (-(input_ids * prompt_mask)) - prompt_mask
        p_embeds = self.prompt_embeddings(prompt_ids) * prompt_mask.float().unsqueeze(-1)

        # a_embeds = None
        # if self.data_hack == "chid":
        #     if not self.is_decoder:
        #         idiom_mask = (input_ids >= self.config.vocab_size).long()
        #         idiom_ids = (input_ids * idiom_mask) - self.config.vocab_size * idiom_mask
        #         a_embeds = self.idiom_fc(self.add_embeds(idiom_ids)) * idiom_mask.float().unsqueeze(-1)

        word_mask = (0 <= input_ids).long()
        word_ids = word_mask * input_ids
        w_embeds = self.word_embeddings(word_ids) * word_mask.float().unsqueeze(-1)

        if p_embeds is not None:
            w_embeds = w_embeds + p_embeds
        # if a_embeds is not None:
        #     w_embeds = w_embeds + a_embeds

        return w_embeds
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0,word_dict=None,word_nums=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeds(input_ids)
            #inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings


        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        #self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        word_dict=None,
        word_nums=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            word_dict=word_dict,
            word_nums=word_nums
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


import transformers
class TEMPBert(BertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()
        #self.word_doc_embedding = AutoModel.from_pretrained('C:\TEMP-lstm\simcse-roberta-base')
        #self.word_doc_tokenizer = AutoTokenizer.from_pretrained('C:\TEMP-lstm\simcse-roberta-base')

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
    ):
        """
        for w in list(word_dict.keys()):
            w_tokenize = self.word_doc_tokenizer(word_dict[w][1],return_tensors='pt').input_ids.to(self.word_doc_embedding.device)
            word_dict[w] = [word_dict[w][0], word_dict[w][1], self.word_doc_embedding(w_tokenize).pooler_output]
        """



        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits,outputs

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class ElectraClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TEMPElectra(ElectraPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: ElectraConfig):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        config.num_labels = 1
        self.classifier = ElectraClassificationHead(config)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        sequence_output = discriminator_hidden_states[0]
        logits = self.classifier(sequence_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPAlbert(AlbertPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: AlbertConfig):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TEMPRoberta(RobertaPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: RobertaConfig):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        config.num_labels = 1
        self.classifier = RobertaClassificationHead(config)
        # self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPXLNet(XLNetPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: XLNetConfig):
        config.num_labels = 1
        super().__init__(config)
        self.xlnet = XLNetModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.classifier = nn.Linear(config.d_model, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[0]
        output = self.sequence_summary(pooled_output)
        logits = self.classifier(output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()


class TEMPXLM(XLMPreTrainedModel):
    EPS = 1e-9

    def __init__(self, config: XLMConfig):
        config.num_labels = 1
        super().__init__(config)
        self.xlnet = XLMModel(config)
        self.sequence_summary = SequenceSummary(config)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.xlnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
        )
        pooled_output = outputs[0]
        output = self.sequence_summary(pooled_output)
        logits = output.view(-1, 1)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()

# class TEMPBart(BartPretrainedModel):
#     EPS = 1e-9
#
#     def __init__(self, config: BartConfig):
#         super().__init__(config)
#         self.electra = BartModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_fct = nn.BCEWithLogitsLoss()
#         self.init_weights()
#
#     def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             head_mask=None,
#             labels=None,
#     ):
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             head_mask=head_mask,
#         )
#         pooled_output = outputs[1]
#         # pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         if labels is not None:
#             return self.loss_fct(logits, labels)
#         return logits
#
#     @classmethod
#     def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
#         loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
#                 neg_score.squeeze().relu().clamp(min=cls.EPS) +
#                 margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
#         return loss.sum()


from transformers.models.deberta_v2 import *

class TEMPDeBerta(DebertaV2PreTrainedModel):
    EPS = 1e-9

    def __init__(self, config):
        super().__init__(config)
        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifiers = nn.Linear(output_dim, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
        self.post_init()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            head_mask=None,
            labels=None,
    ):
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled_output = outputs[0]
        pooled_output = self.pooler(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels is not None:
            return self.loss_fct(logits, labels)
        return logits

    @classmethod
    def margin_loss_fct(cls, pos_score: torch.Tensor, neg_score: torch.Tensor, margin: torch.Tensor):
        loss = (-(pos_score.squeeze().relu().clamp(min=cls.EPS)) +
                neg_score.squeeze().relu().clamp(min=cls.EPS) +
                margin.squeeze().relu().clamp(min=cls.EPS)).clamp(min=0)
        return loss.sum()