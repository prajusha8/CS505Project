# FAKE NEWS CLASSIFICATION with Naive Bayes, LSTM, and BERT Models
This project builds and trains Naive Bayes and LSTM models for fake news classification, and fine-tunes a pretrained BERTForSequenceClassification model.

## Project Structure
.
├── data/
    ├── news_test.csv
│   ├── news_train.csv
│   └── news_val.csv
├── models/
│   ├── lstm.py
│   └── nb.py
├── bert.py
├── dataloader.py
├── lstm.model
├── nb.model
├── README.md
├── requirements.txt
├── train_lstm.py
├── train_nb.py
├── bert.model/ (download it from Google Drive if you would like to use pretrained model, instructions below)

## Installation
To view the requirements needed for this project, you can look at requirements.txt, or install them using:
pip install -r requirements.txt

## BERT Model
If you do not want to retrain the BERT model, you can download and unzip the pretrained BERT model:
https://drive.google.com/file/d/1alCqRRmITTG-k5LyQ7P_uc2dyB8Ae6rN/view?usp=sharing
unzip bert_model.zip 
and place it into the project directory

## Questions
If you have any questions, please reach out at prajusha@bu.edu!