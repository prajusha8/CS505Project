#SYSTEM IMPORTS
import os
import sys
import pickle
import re
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

#PYTHON PROJECT IMPORTS
from models.nb import NaiveBayes


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

def train_model(X_train, y_train, filename):
    model = NaiveBayes()
    model.train(X_train, y_train)
    model.save(filename)
    return model

def main():
    filename = 'nb.model'
    #load training and testing data
    #csv file delimiter is ";" so made sure to change that from default which is ","
    train_data = pd.read_csv('./data/news_train.csv', sep=';').head(10000) #reduce size of training data to 10,000
    test_data = pd.read_csv('./data/news_test.csv', sep=';')

    #clean + tokenize the text column
    X_train = train_data['text'].apply(preprocess_text)
    X_test = test_data['text'].apply(preprocess_text)

    #convert labels to an np array
    y_train = train_data['label'].values
    y_test = test_data['label'].values
    
    #checks to see if the model already exists, and if not it trains a model and saves it as pkl file
    if os.path.exists(filename):
        model = NaiveBayes.load(filename)
    else:
        model = train_model(X_train, y_train, filename)

    #model predictions on test data
    y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
