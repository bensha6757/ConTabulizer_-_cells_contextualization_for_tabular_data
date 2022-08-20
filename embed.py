import pandas as pd
from transformers import T5ForConditionalGeneration, BertModel, BertTokenizer, T5Tokenizer
import torch


class Embedder:
    def __init__(self, t5_for_template_generation, generator_name, encoder_name):
        self.device = self.get_curr_device()
        self.template_generator = T5ForConditionalGeneration.from_pretrained(t5_for_template_generation).to(self.device)
        self.template_generator_tokenizer = T5Tokenizer.from_pretrained(generator_name)
        self.encoder = BertModel.from_pretrained(encoder_name).to(self.device)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(encoder_name)

    def forward(self, table_name, data):
        self.generate_encoding()

    def generate_encoding(self, table_content):
        encodings_by_row = []
        for row in table_content:
            tokenized_row = self.encoder_tokenizer.batch_encode_plus(row)
            last_hidden_state = self.encoder(tokenized_row).last_hidden_state
            entries_encoding = last_hidden_state[:, 0, :].squeeze(1).unsqueeze(0)
            encodings_by_row.append(entries_encoding)
        embedded_input = torch.stack(encodings_by_row, dim=0)
        return embedded_input

    @staticmethod
    def get_curr_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if device == "cuda":
            curr_cuda = torch.cuda.current_device()
            device += ":" + str(curr_cuda)
        return device


