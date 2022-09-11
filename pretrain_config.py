t5_for_generation = 't5-small'
finetuned_t5_for_template_generation = './checkpoints/template_generator'
template_tokenizer_name = 't5-base'
template_encoder_name = 'distilbert-base-uncased'
input_dim = 768
hidden_dim = 512
num_transformer_blocks = 4
heads = 8
row_dim_head = 16
table_dim_head = 16
attn_dropout = 0.1
ff_dropout = 0.1
datasets_dir = 'benchmark-data/benchmark1'
# datasets_dir = 'train-data/csvs'
number_of_records_per_crop = 5
is_shuffle = True
checkpoint_dir = './checkpoints/benchmark1'
is_pretrain = False
