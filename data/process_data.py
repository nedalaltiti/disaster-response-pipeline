# import libraries
import re
import sys
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV 
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): The file path of the messages dataset.
        categories_filepath (str): The file path of the categories dataset.

    Returns:
        pandas.DataFrame: A merged DataFrame containing messages and corresponding categories.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id', how='inner')
    return df


def clean_data(df):
    """
    Clean and preprocess the input DataFrame containing message categories.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a column 'categories' with semicolon-separated category strings.

    Returns:
        pandas.DataFrame: Cleaned DataFrame with individual category columns and numeric values.
    """
    # Select the first row of the categories dataframe
    categories = df["categories"].str.split(pat=";", n=-1, expand=True)

    # select the first row of the categories dataframe
    # Cut the last 2 characters of each category
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])

    # Rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')

    # Drop the duplicates.
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Save a DataFrame to an SQLite database.

    Args:
        df (pandas.DataFrame): The DataFrame to be saved.
        database_filename (str): The desired filename for the SQLite database.

    Returns:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_messages_table', engine, index=False)
 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()