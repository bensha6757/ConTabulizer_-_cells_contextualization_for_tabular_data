import math

import numpy as np
import pandas
import torch
from einops import rearrange
from torch import nn, einsum
from transformers import T5Tokenizer
from transformers.utils import ModelOutput


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
        self.template_generator_tokenizer = T5Tokenizer.from_pretrained('t5-base')

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
        x = x.view(-1, x.shape[-1])
        x = x.unsqueeze(0).to(self.device)

        outputs = ModelOutput()
        outputs["last_hidden_state"] = x
        generation_output = self.t5_model.generate(
            inputs_embeds=x,
            encoder_outputs=outputs,
            num_beams=4,
            max_length=25,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            return_dict_in_generate=True
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
