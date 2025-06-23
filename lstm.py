#SYSTEM IMPORTS
import os
import sys
import torch as pt
import torch.nn as nn
from torch import Tensor
from typing import Type, Tuple
from collections.abc import Sequence, Mapping

LSTMType: Type = Type["LSTM"]

_cd_ = os.path.dirname(os.path.abspath(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_

class LSTM(pt.nn.Module):
    def __init__(self: LSTMType,
                vocab_size: int,
                hidden_size: int=128,
                num_classes: int=2,
                num_epochs: int=20) -> None:

        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size) #embedding layer
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True) #lstm layer
        self.output_layer = nn.Linear(hidden_size, num_classes) #output layer


    def forward(self: LSTMType, x: pt.Tensor) -> pt.Tensor:
        x = self.embedding(x)
        out, (h, c) = self.lstm(x)
        output = self.output_layer(h[-1]) #final hidden state from last lstm layer
        return output

    def train_model(self: LSTMType,
                    train_loader,
                    val_loader,
                    num_epochs: int=20, #experimented with 10 at first, and then defaulted to 20 w/ early stopping
                    lr: float=1e-4) -> None: #started with lr of 1e-5 (too low), and then 1e-3 (too high), so chose 1e-4 as default

        #use Adam optimizer and cross entropy loss function for training lstm         
        o = pt.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        #init variables for early stopping
        best_val_loss = float('inf')
        counter = 0

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                o.zero_grad()
                outputs = self(inputs) #se;f.
                loss = loss_fn(outputs, labels)
                loss.backward()
                o.step()

                train_loss += loss.item()

            #track val. loss and employ early stopping in case val.loss starts to increase again
            #if training loss keeps decreasing while val. loss increases, that's a sign of overfitting to training data
            self.eval()
            val_loss = 0.0
            with pt.no_grad(): #don't track gradients for validation
                for val_inputs, val_labels in val_loader:
                    val_outputs = self(val_inputs)
                    val_loss += loss_fn(val_outputs, val_labels).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0
            else:
                counter += 1
                if counter >= 3:  #if val. loss continues to increase for 3 epochs, then stop training
                    break

            print(f"epoch {epoch+1}/{num_epochs} - train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")


    def predict(self: LSTMType, X_test: pt.Tensor) -> pt.Tensor:
        self.eval()
        with pt.no_grad():
            outputs = self(X_test) #self.forward(X_test)
            _, predicted = pt.max(outputs, 1) #index of the class with the highest score
        return predicted #predicted class