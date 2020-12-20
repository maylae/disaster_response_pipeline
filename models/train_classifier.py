import re
import sys
import nltk
import pandas as pd
import numpy as np
import pickle
import joblib

from nltk import pos_tag
nltk.download(['stopwords', 'wordnet', 'punkt'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, punkt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sqlalchemy import create_engine


def load_data(database_filepath):
    '''
    Load data from database and extract X, y and category names
    :param database_filepath: Path to the database where messages are stored
    :return: messages, categories assigned to messages, category labels
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("categorized_messages", engine)
    X = df["message"]
    Y = df.iloc[:, 5:]
    return X, Y, Y.columns


def tokenize(text):
    '''
    Replace urls in text, tokenize text into words, remove stop words and lemmatize the tokens
    :param text: String to tokenize
    :return: word tokenized and lemmatized text
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Build pipeline for transformation and classification for disaster message classification
    :return: Pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Check performance of machine learning model on test data
    :param model: Estimator to evaluate
    :param X_test: Messages in test set
    :param Y_test: Categories for test set
    :param category_names: Labels for the categories
    :return: F1 score, precision and recall for each category
    '''

    y_pred = model.predict(X_test)
    np.set_printoptions(threshold=1000)

    for i in range(len(category_names)):
        metrics = precision_recall_fscore_support(Y_test[category_names[i]], y_pred[:,i], average="weighted")
        # Support is always None because average parameter is provided
        print("Precision, recall, fscore, support for category", (i+1), category_names[i], metrics)
    return metrics


def save_model(model, model_filepath):
    '''

    :param model: machine learning model
    :param model_filepath: filepath where pickle file should be saved
    :return:
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''

    :return:
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        model.get_params().keys()
        parameters = {
            'clf__estimator__max_features': [10]
        }
        model = GridSearchCV(model, parameters)

        print('Training model...')
        model.fit(X_train, Y_train)
        model = joblib.load("models/classifier.pkl")

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()