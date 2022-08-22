from transformers import T5ForConditionalGeneration, BertModel, BertTokenizer, T5Tokenizer
import torch
from datasets import DatasetHolder
from torch import nn


class Embedder(nn.Module):
    def __init__(self, t5_for_template_generation, generator_name, encoder_name):
        super().__init__()
        self.device = self.get_curr_device()
        self.template_generator = T5ForConditionalGeneration.from_pretrained(t5_for_template_generation).to(self.device)
        self.template_generator_tokenizer = T5Tokenizer.from_pretrained(generator_name)
        self.encoder = BertModel.from_pretrained(encoder_name).to(self.device)
        self.encoder_tokenizer = BertTokenizer.from_pretrained(encoder_name)

    def forward(self, dataset_holder: DatasetHolder):
        table = self.generate_template_sentences(dataset_holder)
        return self.generate_encoding(table)

    def generate_encoding(self, table_content):
        encodings_by_row = []
        for row in table_content:
            tokenized_row = self.encoder_tokenizer.batch_encode_plus(row,
                                                                     padding='max_length',
                                                                     return_tensors='pt',
                                                                     add_special_tokens=True)
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

    def generate_template_sentences(self, dataset_holder: DatasetHolder):
        table = prepare_table_for_template_generator(dataset_holder)
        templates_table = []
        for row in table:
            source_encoding = self.template_generator_tokenizer.batch_encode_plus(row,
                                                                                  padding='max_length',
                                                                                  return_tensors='pt',
                                                                                  add_special_tokens=True)
            generation_output = self.template_generator.generate(
                input_ids=source_encoding['input_ids'].to(self.device),
                attention_mask=source_encoding['attention_mask'].to(self.device),
                num_beams=1,
                max_length=10,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
                use_cache=True,
                return_dict_in_generate=True,
            )
            generated_ids = generation_output.sequences
            preds = [
                self.template_generator_tokenizer.decode(generated_id,
                                                         skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
                for generated_id in generated_ids
            ]
            templates_table.append(preds)

        return templates_table


def prepare_table_for_template_generator(dataset_holder: DatasetHolder):
    row_names = dataset_holder.row_names
    col_names = dataset_holder.col_names
    table_content = dataset_holder.table_content

    for row_idx, row in enumerate(table_content):
        for col_idx, cell_value in enumerate(row):
            table_content[row_idx][col_idx] = \
                row_names[row_idx] + ' # ' + col_names[col_idx] + ' # ' + cell_value + ' # ' + dataset_holder.table_name
    return table_content
