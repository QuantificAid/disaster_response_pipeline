# import system-libraries
import sys
import re

# import other libraries
import pandas as pd
import nltk

# import sqlalchemy-module
from sqlalchemy import create_engine

# imports from nltk-submodules
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# imports from sklearn-submodules
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# downloads from nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True);


def load_data(database_filepath):
    '''
    '''
    
    # read data from sql-db into pandas-dataframe
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, con=engine)

    # extract return values from pandas-Dataframe
    X = df[['message']].values[:, 0]
    # all columns except 'message' are target-labels
    category_names = [column for column in df.columns if column != 'message'] 
    Y = df[category_names].values
    
    return X, Y, category_names


def tokenize(text):
    '''
    '''
    # define stop_words and lemmatizer 
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words in tokens
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''
    '''
    
    model = Pipeline([
        ('vectr', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('rfclf', MultiOutputClassifier(
            RandomForestClassifier(class_weight='balanced', 
                                   n_estimators=20, 
                                   max_features=None)))])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    '''
    
    # Predict and evaluate
    Y_pred = model.predict(X_test)
    scores = classification_report(Y_test, 
                                   Y_pred, 
                                   target_names=category_names)
    
    # print evaluat
    print("Testing scores for MultiOutputClassifier:")
    print(scores)
    
    return None


def save_model(model, model_filepath):
    '''
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))
    
    return None


def main():
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