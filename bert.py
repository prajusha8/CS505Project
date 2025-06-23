from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import os
import sys
import torch
import pandas as pd
import numpy as np

_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

train_data = pd.read_csv("./data/news_train.csv", sep=';').head(10000) #reduce size of training data to 10,000
test_data = pd.read_csv("./data/news_test.csv", sep=';')
val_data = pd.read_csv("./data/news_val.csv", sep=';')

train_data = Dataset.from_pandas(train_data)
test_data = Dataset.from_pandas(test_data)
val_data = Dataset.from_pandas(val_data)

modelpath = "./bert.model"
if os.path.exists(modelpath):
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    model = BertForSequenceClassification.from_pretrained(modelpath)

else:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    train_data = train_data.map(tokenize, batched=True)
    val_data = val_data.map(tokenize, batched=True)

    cols = ['input_ids', 'attention_mask', 'label']
    train_data.set_format(type="torch", columns=cols)
    val_data.set_format(type="torch", columns=cols)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16, #train w/ smaller batch sizes and eval. w/ bigger batch sizes
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data
    )

    trainer.train()

    model.save_pretrained(modelpath)
    tokenizer.save_pretrained(modelpath)

def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
test_data = test_data.map(tokenize, batched=True)
test_data.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])

training_args = TrainingArguments(output_dir="./results", per_device_eval_batch_size=64)
trainer = Trainer(model=model, args=training_args)
results = trainer.predict(test_data)

preds = np.argmax(results.predictions, axis=1)
labels = results.label_ids
print("\nTest Set Classification Report:")
print(classification_report(labels, preds, digits=4))


