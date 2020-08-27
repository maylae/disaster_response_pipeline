import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion

class StartingVerbExtractor (BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        '''

        :param text:
        :return:
        '''
        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        '''

        :param X:
        :param y:
        :return:
        '''
        return self

    def transform(self, X):
        '''

        :param X:
        :return:
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    '''

    :param database_filepath:
    :return:
    '''
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("categorized_messages", engine)
    X = df["message"]
    Y = df.iloc[:, 5:]
    return X, Y, Y.columns


def tokenize(text):
    '''

    :param text:
    :return:
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''

    :return:
    '''
    pipeline = Pipeline([
        ('features', Pipeline([
            ('text_features', Pipeline([
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
                ])),
            ('verb', StartingVerbExtractor())
            ])),
         ('clf', RandomForestClassifier())
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''

    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    '''
    y_pred = model.predict(X_test)
    confusion_mat = confusion_matrix(Y_test, y_pred, labels=category_names)
    accuracy = (y_pred == Y_test).mean()


def save_model(model, model_filepath):
    '''

    :param model:
    :param model_filepath:
    :return:
    '''
    pass


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
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
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