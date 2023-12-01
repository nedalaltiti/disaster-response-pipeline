# import libraries
import re
import pickle
import pandas as pd 

from sqlalchemy import create_engine
import nltk 
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import sys


def load_data(database_filepath):
    """
       Function:
       load data from an SQLite database.

       Args:
        database_filepath (str): The path to the SQLite database file.

       Return:
       X (DataFrame) : Message features dataframe
       Y (DataFrame) : target dataframe
       category (list of str) : target labels list
       """    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster_messages_table', engine)
    X = df['message']
    y = df.iloc[:, 4:]  # Classification label
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
    """
    Function: split text into words and return the root form of the words
    Args:
      text(str): the message
    Return:
      lemm(list of str): a list of the root form of the message words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stop = stopwords.words('english')
    words = [w for w in words if w not in stop]
    
    # lemmatization
    clean_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return clean_tokens


def build_model():
    """
    Build a machine learning pipeline for classifying text messages into multiple categories.

    The pipeline includes:
    - CountVectorizer: Converts text messages into a numerical format by counting the occurrences of each word.
    - TfidfTransformer: Computes TF-IDF (term frequency-inverse document frequency) scores for the words.
    - MultiOutputClassifier with AdaBoostClassifier: Uses decision trees to classify messages into categories.

    Returns:
        GridSearchCV: A grid search cross-validation object to find the best hyperparameters for the model.
    """

    # Create a pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # Create Grid Search Parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 60, 70]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return pipeline
    

def evaluate_model(model, X_test, y_test, category_names):
    """
    Function: Evaluate the model and print the f1 score, precision and recall for each output category of the dataset.
    Args:
        model (object): The trained machine learning model.
        X_test (pandas.Series): Input messages for testing.
        y_test (pandas.DataFrame): True labels for testing.
        category_names (Index): Names of the target categories.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names):
        print('Feature {}: {}'.format(i + 1, col))
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))    


def save_model(model, model_filepath):
    """
    Save a pickle file of the trained model

    Args:
        model (object): The trained machine learning model.
        model_filepath (str): The path where the model will be saved.

    Returns:
        None
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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