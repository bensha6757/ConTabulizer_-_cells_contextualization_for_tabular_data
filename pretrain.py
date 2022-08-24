import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from datasets import DatasetsWrapper
from model.model import ConTabulizer, ConTabulizerForGeneration
from embed import Embedder
from transformers import T5ForConditionalGeneration, Adafactor


class PlConTabulizer(pl.LightningModule):
    def __init__(self, t5_for_generation, finetuned_t5_for_template_generation, template_tokenizer_name, template_encoder_name,
                 dim, nfeats, num_transformer_blocks, heads, row_dim_head, table_dim_head, attn_dropout, ff_dropout,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        embedder = Embedder(finetuned_t5_for_template_generation, template_tokenizer_name, template_encoder_name)
        contabulizer_model = ConTabulizer(dim, nfeats, num_transformer_blocks, heads, row_dim_head,
                                          table_dim_head, attn_dropout, ff_dropout)
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_for_generation)
        self.model = ConTabulizerForGeneration(embedder=embedder, model=contabulizer_model, t5_model=t5_model)

    def forward(self, dataset_holder):
        self.model(dataset_holder)

    def training_step(self, batch):
        dataset_holder = batch
        loss, outputs = self(dataset_holder)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch):
        dataset_holder = batch
        loss, outputs = self(dataset_holder)
        self.log("val loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return Adafactor(
            self.model.parameters(),
            lr=1e-3,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False
        )


class TableDataModule(pl.LightningDataModule):
    def __init__(self, datasets_path, number_of_records_per_crop: int = 10, batch_size: int = 1, is_shuffle=True):
        super(TableDataModule, self).__init__()

        self.train_set = DatasetsWrapper(datasets_path=datasets_path,
                                         number_of_records_per_crop=number_of_records_per_crop,
                                         train_or_val='train',
                                         is_shuffle=is_shuffle)
        self.dev_set = DatasetsWrapper(datasets_path=datasets_path,
                                       number_of_records_per_crop=number_of_records_per_crop,
                                       train_or_val='val',
                                       is_shuffle=is_shuffle)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size)


if __name__ == '__main__':
    datasets_dir = 'train-data/csvs'
    number_of_records_per_crop = 10
    is_shuffle = True

    table_data_module = TableDataModule(datasets_path=datasets_dir,
                                        number_of_records_per_crop=number_of_records_per_crop,
                                        is_shuffle=is_shuffle)
    model = PlConTabulizer(t5_for_generation='t5-small',
                           finetuned_t5_for_template_generation='t5-base',
                           template_tokenizer_name='t5-base',
                           template_encoder_name='bert-base-uncased',
                           dim=768,
                           nfeats=4,
                           num_transformer_blocks=4,
                           heads=8,
                           row_dim_head=16,
                           table_dim_head=16,
                           attn_dropout=0.1,
                           ff_dropout=0.1)

    wandb_logger = WandbLogger(
        name="",
        project="",
        entity="roicohen9"
    )

    val_loss_checkpoint_callback = ModelCheckpoint(monitor="val loss", mode="min")

    gpus = 1
    train_strategy = 'ddp_sharded'

    trainer = pl.Trainer(
        max_epochs=6,
        gpus=gpus,
        num_nodes=1,
        strategy=train_strategy,
        # precision=16,
        logger=wandb_logger,
        callbacks=[val_loss_checkpoint_callback]
    )
    trainer.fit(model, table_data_module)
