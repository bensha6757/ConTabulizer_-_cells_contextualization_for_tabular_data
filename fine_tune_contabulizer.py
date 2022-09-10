import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import fairscale
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import wandb
import torch

from datasets import DatasetsWrapper
from model.model import ConTabulizer, ConTabulizerForGeneration
from embed import Embedder
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor

from pretrain import TableDataModule, PlConTabulizer
from pretrain_config import t5_for_generation, finetuned_t5_for_template_generation, template_tokenizer_name, \
    template_encoder_name, input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head, table_dim_head, \
    attn_dropout, ff_dropout


if __name__ == '__main__':
    datasets_dir = 'benchmark-data/benchmark1'
    number_of_records_per_crop = 5
    is_shuffle = True
    wandb.init(project="contabulizer_fine_tuning")

    table_data_module = TableDataModule(datasets_path=datasets_dir,
                                        number_of_records_per_crop=number_of_records_per_crop,
                                        is_shuffle=is_shuffle,
                                        is_pretrain=False)
    model = PlConTabulizer(t5_for_generation=t5_for_generation,
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
                           dev_set=table_data_module.dev_set)

    wandb_logger = WandbLogger(
        name=f"{t5_for_generation}-{finetuned_t5_for_template_generation}-{template_tokenizer_name}-"
             f"{template_encoder_name}-{input_dim}-{hidden_dim}-{num_transformer_blocks}-{heads}-{row_dim_head}-"
             f"{table_dim_head}-{attn_dropout}-{ff_dropout}",
        project="ConTabulizer_benchmarking",
        entity="roicohen9"
    )

    val_loss_checkpoint_callback = ModelCheckpoint(monitor="val loss", mode="min")

    gpus = 1
    train_strategy = 'ddp_sharded'

    trainer = pl.Trainer(
        max_epochs=10,
        gpus=gpus,
        num_nodes=1,
        strategy=train_strategy,
        # precision=16,
        logger=wandb_logger,
        callbacks=[val_loss_checkpoint_callback],
        default_root_dir='./checkpoints/benchmarks'
    )
    trainer.fit(model, table_data_module)
