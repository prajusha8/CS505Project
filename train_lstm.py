#SYSTEM IMPORTS
import os
import sys
import re
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from collections import Counter

#PYTHON PROJECT IMPORTS
from models.lstm import LSTM
from dataloader import NewsDataset

#Constants declared in this module
START_TOKEN: str = "<BOS>"
END_TOKEN: str = "<EOS>"
UNK_TOKEN: str = "<UNK>"
PAD_TOKEN: str = "<PAD>"

_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) #using regex to remove random chars and replace them w/ empty string
    text = text.lower()
    tokens = text.split()
    return tokens

def train_lstm(X_train, y_train, X_val, y_val, vocab, batch_size=32, num_epochs=20):

    train_dataset = NewsDataset(X_train, y_train, vocab)
    val_dataset = NewsDataset(X_val, y_val, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle data for training set
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) #don't shuffle for validation set

    model = LSTM(vocab_size=len(vocab), hidden_size=128, num_classes=2)
    model.train_model(train_loader, val_loader)

    torch.save(model.state_dict(), "lstm.model")

    return model

def evaluate(X_test, y_test, vocab, batch_size=32):
    test_dataset = NewsDataset(X_test, y_test, vocab)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = LSTM(vocab_size=len(vocab))
    model.load_state_dict(torch.load("lstm.model"))
    model.eval()

    predictions = []
    labels = []

    for inputs, labs in test_loader:
        outputs = model.predict(inputs)
        predictions.extend(outputs.numpy())
        labels.extend(labs.numpy())

    print(f"Accuracy: {accuracy_score(labels, predictions):.4f}")
    print("Classification Report:")
    print(classification_report(labels, predictions))


def main():
    filename = "lstm.model"

    train_data = pd.read_csv('./data/news_train.csv', sep=';').head(10000) #reduce size of training data to 10,000
    test_data = pd.read_csv('./data/news_test.csv', sep=';')
    val_data = pd.read_csv('./data/news_val.csv', sep=';')

    X_train = train_data['text'].apply(preprocess_text)
    X_test = test_data['text'].apply(preprocess_text)
    X_val = val_data['text'].apply(preprocess_text)
    
    y_train = train_data['label'].values
    y_test = test_data['label'].values
    y_val = val_data['label'].values


    #maps words to indices
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    i = 2
    for article in X_train:
        for word in article:
            if word not in vocab:
                vocab[word] = i
                i += 1
        
    if os.path.exists(filename):
        model = LSTM(vocab_size=len(vocab), hidden_size=128, num_classes=2)
        model.load_state_dict(torch.load(filename))
    else:
        model = train_lstm(X_train, y_train, X_val, y_val, vocab)

    evaluate(X_test, y_test, vocab)

if __name__ == "__main__":
    main()

