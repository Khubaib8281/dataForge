import sqlite3

def init_db(df):
    conn = sqlite3.connect(":memory:") # in-memory database
    df.to_sql("df_cleaned", conn, index=False, if_exists = "replace")
    return conn     