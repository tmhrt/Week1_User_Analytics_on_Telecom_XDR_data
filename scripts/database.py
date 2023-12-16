import matplotlib as plt
import seaborn as sns
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

def connect_to_database(connection_params: dict):
    """
    Connects to the PostgreSQL database.
    paramters:
        connection_params is a dictionary that define the following:
        {
            'dbname': 'your_database_name',
            'user': 'your_username',
            'password': 'your_password',
            'host': 'your_host',
            'port': 'your_port'
            }
    """
    try:
        connection = psycopg2.connect(**connection_params)
        return connection
    except psycopg2.Error as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None

def read_table_to_dataframe(table_name, connection_params):
    """
    Reads a PostgreSQL table into a pandas dataframe.
    """
    connection = connect_to_database(connection_params)
    if connection:
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql_query(query, connection)
        connection.close()
        return df
    else:
        print("Error, no connection detected!")
        return None

def write_dataframe_to_table(df, table_name, connection_params):
    """
    Writes a pandas dataframe to a new table in the PostgreSQL database.
    """
    engine = create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}")
    df.to_sql(table_name, engine, index=False, if_exists='replace')
    print(f"Dataframe successfully written to the '{table_name}' table.")

def update_table_by_appending(df, table_name, connection_params):
    """
    Appends a pandas dataframe to an existing PostgreSQL table.
    """
    engine = create_engine(f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}")
    df.to_sql(table_name, engine, index=False, if_exists='append')
    print(f"Dataframe successfully appended to the '{table_name}' table.")

def delete_table(table_name, connection_params):
    """
    Deletes a table from the PostgreSQL database.
    """
    connection = connect_to_database(connection_params)
    if connection:
        cursor = connection.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        connection.commit()
        connection.close()
        print(f"Table '{table_name}' successfully deleted.")
    else:
        print("Error: Unable to connect to the database.")


def load_data_to_df():
    connection_parameters = {
    'dbname': 'telecom_db',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
    }
    df_read = read_table_to_dataframe('xdr_data', connection_parameters)
    df_read.to_pickle("data/telecom_xdr.pkl")
    
###################################PLOTTING FUNCTIONS###################################

def plot_hist(df:pd.DataFrame, column:str, color:str, title:str)->None:
    sns.displot(data=df, x=column, color=color, kde=True, height=7, aspect=2)
    plt.title(title, size=20, fontweight='bold')
    plt.yscale("log")
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_count(df:pd.DataFrame, column:str, title:str, cutt_off=None) -> None:
    if (cutt_off == None) or (cutt_off > df[column].nunique()):
        cutt_off = df[column].nunique()
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=column, order = df[column].value_counts().iloc[:cutt_off].index)
    plt.title(title, size=20, fontweight='bold')
    fig = plt.gcf()
    plt.tick_params(axis='x', rotation=90)
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_count_multilevel(df:pd.DataFrame, main_group:str, sub_group:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df, x=main_group, hue=sub_group, order = df[main_group].value_counts().index)
    plt.title(title, size=20, fontweight='bold')
    fig = plt.gcf()
    plt.tick_params(axis='x', rotation=90)
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_bar(df:pd.DataFrame, x_col:str, y_col:str, title:str, xlabel:str, ylabel:str)->None:
    plt.figure(figsize=(12, 7))
    sns.barplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20, fontweight='bold')
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_heatmap(df:pd.DataFrame, title:str, cbar=False)->None:
    plt.figure(figsize=(12, 7))
    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar )
    plt.title(title, size=18, fontweight='bold')
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col)
    plt.title(title, size=20, fontweight='bold')
    plt.xticks(rotation=75, fontsize=14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str) -> None:
    plt.figure(figsize=(12, 7))
    sns.boxplot(data = df, x=x_col, y=y_col)
    plt.title(title, size=20, fontweight='bold')
    plt.xticks(rotation=75, fontsize=14)
    plt.yticks( fontsize=14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+f'Multivariate {x_col} vs {y_col}'+'.png', dpi=100)

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data = df, x=x_col, y=y_col, hue=hue, style=style)
    plt.title(title, size=20, fontweight='bold')
    plt.xticks(fontsize=14)
    plt.yticks( fontsize=14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)

def plot_bivariate_pair(df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(24, 14))
    sns.pairplot(df, diag_kind = 'kde', height=4)#hue = 'link', 
    plt.title(title, size=20, fontweight='bold')
    fig = plt.gcf()
    plt.show()
    fig.savefig('../plots/'+title+'.png', dpi=100)
'''
# Example 2: Write a pandas dataframe to a new table
df_write = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
write_dataframe_to_table(df_write, 'new_table', connection_parameters)

# Example 3: Update a table by appending a dataframe
df_append = pd.DataFrame({'col1': [4, 5], 'col2': ['d', 'e']})
update_table_by_appending(df_append, 'new_table', connection_parameters)

# Example 4: Delete a table
delete_table('new_table', connection_parameters)
'''