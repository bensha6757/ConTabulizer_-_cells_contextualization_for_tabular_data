import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import fairscale
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import numpy as np
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer, Adafactor


class T5TemplateGeneration(pl.LightningModule):
    def __init__(self, t5_model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

    def forward(self, x):
        output = self.model(x)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        loss, outputs = self(batch)
        self.log("train loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, outputs = self(batch)
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
    def __init__(self, train_df, val_df, model_params):
        super(TableDataModule, self).__init__()

        tokenizer = T5Tokenizer.from_pretrained(model_params['MODEL'])
        self.train_set = RowColValue2SentenceDataset(dataframe=train_df,
                                                     tokenizer=tokenizer,
                                                     source_len=model_params['MAX_SOURCE_TEXT_LENGTH'],
                                                     target_len=model_params['MAX_TARGET_TEXT_LENGTH'],
                                                     source_text="text",
                                                     target_text="headlines")
        self.dev_set = RowColValue2SentenceDataset(dataframe=val_df,
                                                   tokenizer=tokenizer,
                                                   source_len=model_params['MAX_SOURCE_TEXT_LENGTH'],
                                                   target_len=model_params['MAX_TARGET_TEXT_LENGTH'],
                                                   source_text="text",
                                                   target_text="headlines")
        self.train_batch_size = model_params['TRAIN_BATCH_SIZE']
        self.val_batch_size = model_params['TRAIN_BATCH_SIZE']

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.val_batch_size)


class RowColValue2SentenceDataset(Dataset):

    def __init__(
            self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = source["input_ids"].squeeze()
        attention_mask = source["attention_mask"].squeeze()
        target_input_ids = target["input_ids"].squeeze()

        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "labels": target_input_ids.to(dtype=torch.long),
        }


if __name__ == '__main__':
    # wandb.init(project="contabulizer")
    model_params = {
        "MODEL": "t5-base",  # model_type: t5-base/t5-large
        "TRAIN_BATCH_SIZE": 8,  # training batch size
        "VALID_BATCH_SIZE": 8,  # validation batch size
        "TRAIN_EPOCHS": 10,  # number of training epochs
        "VAL_EPOCHS": 1,  # number of validation epochs
        "LEARNING_RATE": 1e-4,  # learning rate
        "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
        "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
        "SEED": 42,  # set seed for reproducibility
    }

    data_path = "data.csv"

    data_df = pd.read_csv(data_path)

    data_df = data_df[['text', 'headlines']]

    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    train_size = 0.8
    train_dataset = data_df.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = data_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    table_data_module = TableDataModule(train_df=train_dataset,
                                        val_df=val_dataset,
                                        model_params=model_params)
    model = T5TemplateGeneration(t5_model_name=model_params["MODEL"])

    wandb_logger = WandbLogger(
        name=f"",
        project="T5TemplateGeneration",
        entity="bensha"
    )

    val_loss_checkpoint_callback = ModelCheckpoint(monitor="val loss", mode="min")

    gpus = 0
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
