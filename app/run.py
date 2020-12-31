import json
import re

import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from plotly.graph_objs import Bar, Heatmap
# from sklearn.externals \
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")
print(model)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    category_names = df.columns[2:]
    category_counts = df[category_names].sum()
    print(category_names)
    print(category_counts)
    genre_names = list(genre_counts.index)
    number_of_labels = pd.DataFrame(df[2:].sum(axis=1).value_counts(), columns=['count'])
    heatmap_counts = np.zeros((len(category_names), len(category_names)))
    for i in range(len(category_names)):
        for j in range(len(category_names)):
            if i >= j:  # symmetric matrix, so we only need to compute one half and copy it to the other one
                # print(i, j, category_names[i], category_names[j])
                heatmap_counts[(i, j)] = (df[category_names[i]] * df[category_names[j]]).sum() / len(
                    df[category_names[i]])
                heatmap_counts[(j, i)] = heatmap_counts[(i, j)]
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=number_of_labels.index,
                    y=number_of_labels['count']
                )
            ],

            'layout': {
                'title': 'Number of labels for messages',
                'yaxis': {
                    'title': "Number of labels"
                },
                'xaxis': {
                    'title': "Number of messages with this number of labels"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names,
                    z=heatmap_counts
                )
            ],
            'layout': {
                'title': 'Heatmap of Message Category classifications',
                'height': 800,
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')
    print("query", query)
    # use model to predict classification for query
    prediction = model.predict([query])
    print("prediction", prediction)
    print("model", model)
    classification_labels = prediction[0]
    print("classification_labels", classification_labels)
    classification_results = dict(zip(df.columns[2:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='localhost', port=3001, debug=True)


if __name__ == '__main__':
    main()
