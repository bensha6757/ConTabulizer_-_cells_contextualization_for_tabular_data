import math

import numpy as np
import pandas
import torch
from einops import rearrange
from torch import nn, einsum


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, input_dim, output_dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mult, output_dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class ConTabulizer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head, table_dim_head, attn_dropout,
                 ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(PreNorm(input_dim, FeedForward(input_dim, hidden_dim)))
        for _ in range(num_transformer_blocks):
            self.layers.append(nn.ModuleList([
                PreNorm(hidden_dim, Residual(Attention(hidden_dim, heads=heads, dim_head=row_dim_head, dropout=attn_dropout))),
                PreNorm(hidden_dim, Residual(FeedForward(hidden_dim, hidden_dim, dropout=ff_dropout))),
                PreNorm(hidden_dim,
                        Residual(Attention(hidden_dim, heads=heads, dim_head=table_dim_head, dropout=attn_dropout))),
                PreNorm(hidden_dim, Residual(FeedForward(hidden_dim, hidden_dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):
        first_layer_for_changing_dim = self.layers[0]
        x = first_layer_for_changing_dim(x)
        x_shape = x.shape
        for row_attn, ff1, table_attn, ff2 in self.layers[1:]:
            x = row_attn(x)
            x = ff1(x)
            x = x.view(-1, x.shape[-1]).unsqueeze(0)
            x = table_attn(x)
            x = x.view(*x_shape)
            x = ff2(x)
        return x


class ConTabulizerForGeneration(nn.Module):
    def __init__(self, embedder, model, t5_model, tokenizer):
        super().__init__()
        self.device = self.get_curr_device()
        self.embedder = embedder
        self.model = model
        self.t5_model = t5_model
        self.tokenizer = tokenizer

    def forward(self, dataset_holder_dict):
        x = self.embedder(dataset_holder_dict)
        x = self.model(x)
        tokenized_labels = self.tokenizer(dataset_holder_dict['label'], padding=True, return_tensors='pt').input_ids.to(self.device)
        x = x.view(-1, x.shape[-1])
        x = x.unsqueeze(0).to(self.device)
        return self.t5_model(inputs_embeds=x, encoder_outputs=[x], labels=tokenized_labels)

    def generate(self, dataset_holder_dict):
        x = self.embedder(dataset_holder_dict)
        x = self.model(x)
        tokenized_labels = self.tokenizer(dataset_holder_dict['label'], padding=True, return_tensors='pt').input_ids.to(self.device)
        x = x.view(-1, x.shape[-1])
        x = x.unsqueeze(0).to(self.device)

        generation_output = self.template_generator.generate(
            inputs_embeds=x,
            encoder_outputs=[x],
            num_beams=4,
            max_length=25,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            return_dict_in_generate=True,
        )
        generated_ids = generation_output.sequences.to(self.device)
        preds = [
            self.template_generator_tokenizer.decode(generated_id,
                                                     skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)
            for generated_id in generated_ids
        ]

        return preds

    @staticmethod
    def get_curr_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if device == "cuda":
            curr_cuda = torch.cuda.current_device()
            device += ":" + str(curr_cuda)
        return device

# class BertAttention(nn.Module):
#     def __init__(self, config, position_embedding_type=None):
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
#             raise ValueError(
#                 f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
#                 f"heads ({config.num_attention_heads})"
#             )
#
#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size
#
#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)
#
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.position_embedding_type = position_embedding_type or getattr(
#             config, "position_embedding_type", "absolute"
#         )
#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             self.max_position_embeddings = config.max_position_embeddings
#             self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
#
#         self.is_decoder = config.is_decoder
#
#     def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(new_x_shape)
#         return x.permute(0, 2, 1, 3)
#
#     def forward(self, hidden_states: torch.Tensor):
#
#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))
#         query_layer = self.transpose_for_scores(self.query(hidden_states))
#
#         if self.is_decoder:
#             # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
#             # Further calls to cross_attention layer can then reuse all cross-attention
#             # key/value_states (first "if" case)
#             # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
#             # all previous decoder key/value_states. Further calls to uni-directional self-attention
#             # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
#             # if encoder bi-directional self-attention `past_key_value` is always `None`
#             past_key_value = (key_layer, value_layer)
#
#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
#
#         if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
#             seq_length = hidden_states.size()[1]
#             position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
#             position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
#             distance = position_ids_l - position_ids_r
#             positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
#             positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility
#
#             if self.position_embedding_type == "relative_key":
#                 relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores
#             elif self.position_embedding_type == "relative_key_query":
#                 relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
#                 relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
#                 attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
#
#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#
#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)
#
#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)
#
#         context_layer = torch.matmul(attention_probs, value_layer)
#
#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(new_context_layer_shape)
#
#         outputs = (context_layer, attention_probs)
#
#         if self.is_decoder:
#             outputs = outputs + (past_key_value,)
#         return outputs
