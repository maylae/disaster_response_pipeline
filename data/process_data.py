import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load csv files containing disaster messages and categories
    :param messages_filepath: path to disaster messages csv file
    :param categories_filepath: path to disaster categories csv file
    :return: dataframe containing messages joined with categories
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories, on='id', how='inner', rsuffix='_cat')
    return df


def clean_data(df):
    '''
    Extract single columns from categories column, transform them into binary columns and drop duplicates.
    :param df: Dataframe of messages with assigned categories
    :return: Cleaned and deduplicated dataframe with single categories as columns
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[:0, :]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda col: col.apply(lambda y: y[0:-2]), axis=1).values[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to binary
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop(['categories', 'id', 'id_cat', 'original'], axis=1, inplace=True)
    # convert to binary
    categories = categories.applymap(lambda col: col > 0)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    '''
    Saves dataframe to database table. Replaces tables if exists.
    :param df: Dataframe to store
    :param database_filename: Filename of database to save dataframe content in
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('categorized_messages', engine, index=False, if_exists = 'replace')


def main():
    '''
    Runs ETL process
    '''
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