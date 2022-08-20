import pandas as pd
from transformers import T5ForConditionalGeneration, BertModel

class Embedder:
    def __init__(self, t5_for_template_generation, encoder_name):
        self.template_generator = T5ForConditionalGeneration.from_pretrained(t5_for_template_generation)
        self.encoder = BertModel.from_pretrained(encoder_name)

    def forward(self, table_name, data):


