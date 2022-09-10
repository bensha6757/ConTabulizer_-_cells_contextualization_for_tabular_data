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
from pretrain_config import t5_for_generation, finetuned_t5_for_template_generation, template_tokenizer_name, \
    template_encoder_name, input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head, table_dim_head, \
    attn_dropout, ff_dropout, datasets_dir, number_of_records_per_crop, is_shuffle, checkpoint_dir


class PlConTabulizer(pl.LightningModule):
    def __init__(self, t5_for_generation, finetuned_t5_for_template_generation, template_tokenizer_name, template_encoder_name,
                 input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head, table_dim_head, attn_dropout, ff_dropout,
                 dev_set, *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedder = Embedder(finetuned_t5_for_template_generation, template_tokenizer_name, template_encoder_name).to(self.device)
        contabulizer_model = ConTabulizer(input_dim, hidden_dim, num_transformer_blocks, heads, row_dim_head,
                                          table_dim_head, attn_dropout, ff_dropout).to(self.device)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_for_generation).to(self.device)
        t5_tokenizer = T5Tokenizer.from_pretrained(t5_for_generation)
        self.model = ConTabulizerForGeneration(embedder=embedder, model=contabulizer_model,
                                               t5_model=t5_model, tokenizer=t5_tokenizer)
        self.dev_set = dev_set

    def forward(self, dataset_holder_dict):
        output = self.model(dataset_holder_dict)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        dataset_holder_dict = batch
        loss, outputs = self(dataset_holder_dict)
        self.log("train loss", loss, prog_bar=True, logger=True, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        dataset_holder_dict = batch
        loss, outputs = self(dataset_holder_dict)
        self.log("val loss", loss, prog_bar=True, logger=True, batch_size=1)
        return loss

    def configure_optimizers(self):
        return Adafactor(
            self.model.parameters(),
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )

    def validation_step_end(self, outputs):
        em, f1 = self.compute_dev_acc(self.dev_set)
        self.log("val_em", em, prog_bar=True, logger=True)
        self.log("val_f1", f1, prog_bar=True, logger=True)

    def compute_dev_acc(self, dev_set):
        exact_answers = 0
        avg_precision = 0
        avg_recall = 0
        for i, example in enumerate(dev_set):
            label = example['label']
            pred = self.model.generate(example)[0]

            if pred == label:
                exact_answers += 1

            precision = 0
            for p in pred:
                if p in label:
                    precision += 1
            precision /= len(pred)
            avg_precision += precision

            recall = 0
            for g in label:
                if g in pred:
                    recall += 1
            recall /= len(label)
            avg_recall += recall

        exact_match = exact_answers / len(dev_set)
        avg_precision /= len(dev_set)
        avg_recall /= len(dev_set)
        avg_f1 = 2 * ((avg_precision * avg_recall) / (avg_precision + avg_recall))
        return exact_match, avg_f1


class TableDataModule(pl.LightningDataModule):
    def __init__(self, datasets_path, number_of_records_per_crop: int = 10, batch_size: int = 1, is_shuffle=True, is_pretrain=True):
        super(TableDataModule, self).__init__()

        self.train_set = DatasetsWrapper(datasets_path=datasets_path,
                                         number_of_records_per_crop=number_of_records_per_crop,
                                         train_or_val='train',
                                         is_shuffle=is_shuffle,
                                         is_pretrain=is_pretrain)
        self.dev_set = DatasetsWrapper(datasets_path=datasets_path,
                                       number_of_records_per_crop=number_of_records_per_crop,
                                       train_or_val='val',
                                       is_shuffle=is_shuffle,
                                       is_pretrain=is_pretrain)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size)


if __name__ == '__main__':
    wandb.init(project="contabulizer")

    table_data_module = TableDataModule(datasets_path=datasets_dir,
                                        number_of_records_per_crop=number_of_records_per_crop,
                                        is_shuffle=is_shuffle,
                                        is_pretrain=True)
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
                           dev_set=DataLoader(table_data_module.dev_set, batch_size=1))

    wandb_logger = WandbLogger(
        name=f"{t5_for_generation}-{finetuned_t5_for_template_generation}-{template_tokenizer_name}-"
             f"{template_encoder_name}-{input_dim}-{hidden_dim}-{num_transformer_blocks}-{heads}-{row_dim_head}-"
             f"{table_dim_head}-{attn_dropout}-{ff_dropout}",
        project="ConTabulizer",
        entity="roicohen9"
    )

    val_loss_checkpoint_callback = ModelCheckpoint(monitor="val loss", mode="min")

    gpus = 0
    train_strategy = 'ddp_sharded'

    trainer = pl.Trainer(
        max_epochs=10,
        gpus=gpus,
        num_nodes=1,
        strategy=train_strategy,
        # precision=16,
        logger=wandb_logger,
        callbacks=[val_loss_checkpoint_callback],
        default_root_dir=checkpoint_dir
    )
    trainer.fit(model, table_data_module)
