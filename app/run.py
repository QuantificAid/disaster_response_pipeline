import re
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('data/DisasterResponse.db', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    related_counts = df.groupby('related').count()['message']
    related_names = [str(item) for item in related_counts.index]
    
    issues_counts = df[df.related == 1][df.columns[2:]].sum()
    issues_names = df.columns[2:]
    
    cat_cols = df.columns[1:]
    cats = df[cat_cols]
    interrelations = pd.concat([cats[cats[col] == 1].mean() for col in cat_cols], axis=1).fillna(0)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=related_names,
                    y=related_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Messages Related to Disasters',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related (1 = True, 0 = False)"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=issues_names,
                    y=issues_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Issues of Disaster Related Messages',
                'xaxis': {
                    'title': "Issue Category"
                },
                'yaxis': {
                    'title': "Count, Given Message Is Disaster Related"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    z=interrelations,
                    x=cat_cols,
                    y=cat_cols
                )
            ],
            
            'layout': {
                'title': 'Interrelations of Categories',
                'xaxis': {
                    'title': "Category"
                },
                'yaxis': {
                    'title': "Category (only being 1=True)"
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

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()