import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Read csv-files at 'messages_filepath' and 'categories_filepath' and 
    return 'df' as pandas dataframe merged on 'id' in messages and categories.
    
    csv's both must have an 'id'-column, messages at 'messages_filepath' must have a
    'message' column, categories at 'categories_filepath' must have a 'categories' column.
    
    Parameters:
        message_filepath as string
        categories_filepath as string
    Returns:
        df as pandas-Dataframe
    '''
    
    try:
        messages = pd.read_csv(messages_filepath)
    except:
        print("No appropriate csv found at 'message_filepath'")
        return None
        
    try:
        categories = pd.read_csv(categories_filepath)
    except:
        print("No appropriate csv found at 'categories_filepath'")
        return None
        
    assert 'id' in messages.columns, "'id'-column missing in csv at 'messages_filepath'"
    assert 'id' in categories.columns, "'id'-column missing in csv at 'categories_filepath'"
    assert 'message' in messages.columns, "'message'-column missing in csv at 'messages_filepath'"
    assert 'categories' in categories.columns, "'categories'-column missing in csv at 'categories_filepath'"
    
    df = pd.merge(messages, categories, on='id')
    
    return df


def clean_data(df):
    '''
    Clean pandas-Dataframe 'df' and returns 'clean_df' 
    
    'df' must have 'categories'-column with mapping to categories in the format
    of 'category-x' with x in [0, 1] and category from categories each separated
    by ';'. 'df' must also include 'message'-column. 'df' can have duplicates.
    
    Parameters:
        df as pandas-Dataframe
    Returns:
        clean_df as pandas-Dataframe
    '''
    
    # extract messages (for later creating nlp-features)
    messages_df = df[['message']]
    
    # extract category columns being 1, 0 from categories-column
    categories_df = df['categories'].str.split(";", expand=True)
    categories_colnames = [item[:-2] for item in categories_df.loc[0]]
    categories_df.columns = categories_colnames
    for column in categories_df:
        categories_df[column] = categories_df[column].apply(lambda x: x[-1]).astype(int)
    
    # assert structure of columns to be matched
    categories = ['related', 
                  
                  # all below only 1 if related 1
                  
                  'request', # if 1 offer not 1
                  'offer', # if 1 request not 1
                  
                  'aid_related', 
                  'medical_help', # only 1 if aid_related 1 
                  'medical_products', # only 1 if aid_related 1  
                  'search_and_rescue', # only 1 if aid_related 1 
                  'security', 'military', # only 1 if aid_related 1 
                  'child_alone', # only 1 if aid_related 1, but here all zero
                  'water', # only 1 if aid_related 1  
                  'food', # only 1 if aid_related 1  
                  'shelter', # only 1 if aid_related 1 
                  'clothing', # only 1 if aid_related 1  
                  'money', # only 1 if aid_related 1  
                  'missing_people', # only 1 if aid_related 1  
                  'refugees', # only 1 if aid_related 1  
                  'death', # only 1 if aid_related 1  
                  'other_aid', # only 1 if aid_related 1 
                  
                  'infrastructure_related', 
                  
                  'transport', # ambigous
                  'buildings', # ambigous
                  'electricity', # ambigous
                  'tools', # ambigous
                  
                  'hospitals', # only 1 if infrastructure_related 1
                  'shops', # only 1 if infrastructure_related 1
                  'aid_centers', # only 1 if infrastructure_related 1
                  'other_infrastructure', # only 1 if infrastructure_related 1
                  
                  'weather_related', 
                  'floods', # only 1 if weather_related 1  
                  'storm', # only 1 if weather_related 1  
                  'fire', # only 1 if weather_related 1  
                  'earthquake', # only 1 if weather_related 1 
                  'cold', # only 1 if weather_related 1 
                  'other_weather', # only 1 if weather_related 1 
                  
                  'direct_report'
                 ]
    assert categories_colnames == categories, "categories do not match defined schema"
    
    # concat dfs and drop duplicates as well as (possible) nans
    clean_df = (
        pd.concat([
            messages_df, 
            categories_df], 
            axis=1)
        .drop_duplicates()
        .dropna()
    )
    
    # drop columns containing 2 as value (being the case for 'related')
    clean_df.drop(
        clean_df[(clean_df[categories_colnames] == 2).any(axis=1) == True]
        .index, 
        inplace=True)
    
    return clean_df


def save_data(df, database_filename):
    '''
    Save data in pandas-dataframe df as sql-database
    with database_filename
    
    Parameters:
        df: pandas dataframe
        database_filename: path to sql-database
        
    Returns:
        None
    '''
    
    # assert input types
    assert isinstance(df, pd.DataFrame), 'no pandas dataframe for first argument'
    assert isinstance(database_filename, str), 'no string as second argument'
    
    # create/replace database
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')
    
    return None  


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