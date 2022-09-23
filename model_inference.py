from transformers import T5Tokenizer

from datasets import DatasetsWrapper
from pretrain import PlConTabulizer
from pretrain_config import t5_for_generation, finetuned_t5_for_template_generation, template_tokenizer_name, \
    template_encoder_name, input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head, table_dim_head, \
    attn_dropout, ff_dropout, datasets_dir, number_of_records_per_crop, is_shuffle, checkpoint_dir, is_pretrain


if __name__ == '__main__':

    ckpt_path = './checkpoints/t5-small-./checkpoints/' \
                'template_generator-t5-base-distilbert-base-uncased-768-512-4-8-16-16-0.1-0.1/' \
                'version_None/checkpoints/epoch=0-step=16754.ckpt'

    model = PlConTabulizer.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        t5_for_generation=t5_for_generation,
        finetuned_t5_for_template_generation=finetuned_t5_for_template_generation,
        template_tokenizer_name=template_tokenizer_name,
        template_encoder_name=template_encoder_name,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_transformer_blocks=num_transformer_blocks,
        heads=heads,
        row_dim_head=row_dim_head,
        table_dim_head=table_dim_head,
        attn_dropout=attn_dropout,
        ff_dropout=ff_dropout,
        dev_set=''
    )
    contabulizer = model.model
    # contabulizer.save_pretrained('../checkpoints/pretrained_contabulizer')

    data_wrapper = DatasetsWrapper(datasets_path='inference_data',
                                   number_of_records_per_crop=5,
                                   train_or_val='train',
                                   is_shuffle=True,
                                   is_pretrain=True)

    for i, dataset_holder_dict in enumerate(data_wrapper):
        generation_output = contabulizer.generate(dataset_holder_dict)
        print(f"pred: {generation_output[0]}\nlabel: {dataset_holder_dict['label'][0]}\n")
