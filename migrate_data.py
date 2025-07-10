import os
import sqlite3
from sqlalchemy import create_engine, text
import sys

# --- CONFIGURATION ---
# Your local SQLite database file
SQLITE_DB_FILE = 'master.db'
# The table you want to migrate
TABLE_NAME = 'content_repo'
# --- END CONFIGURATION ---

def migrate_data():
    """
    Connects to a local SQLite database and a remote PostgreSQL database,
    and copies all data from a specified table from SQLite to PostgreSQL.
    """
    print("--- Starting Data Migration ---")

    # 1. Get the PostgreSQL connection string from an environment variable
    pg_conn_string = os.environ.get('DATABASE_URL')
    if not pg_conn_string:
        print("\nERROR: DATABASE_URL environment variable not set.")
        print("Please set it before running the script.")
        print("Example: export DATABASE_URL='postgres://user:pass@host/db'")
        sys.exit(1) # Exit the script with an error code

    # 2. Check if the local SQLite database exists
    if not os.path.exists(SQLITE_DB_FILE):
        print(f"\nERROR: SQLite database file '{SQLITE_DB_FILE}' not found.")
        print("Please make sure this script is in the same directory as your database.")
        sys.exit(1)

    try:
        # 3. Connect to both databases
        print("Connecting to databases...")
        sqlite_conn = sqlite3.connect(SQLITE_DB_FILE)
        sqlite_cursor = sqlite_conn.cursor()

        pg_engine = create_engine(pg_conn_string)
        pg_conn = pg_engine.connect()
        print("Connections successful.")

        # 4. Fetch all data from the SQLite table
        print(f"Fetching data from SQLite table: {TABLE_NAME}...")
        sqlite_cursor.execute(f"SELECT * FROM {TABLE_NAME}")
        rows = sqlite_cursor.fetchall()
        
        if not rows:
            print("Warning: No rows found in the SQLite database to migrate.")
            return

        print(f"Found {len(rows)} rows to migrate.")

        # Get column names from SQLite
        column_names = [description[0] for description in sqlite_cursor.description]
        # Ensure column names are double-quoted for PostgreSQL
        quoted_column_names = [f'"{col}"' for col in column_names]

        # 5. Clear the destination table in PostgreSQL to avoid duplicates
        print(f"Clearing existing data from PostgreSQL table: {TABLE_NAME}...")
        pg_conn.execute(text(f'TRUNCATE TABLE "{TABLE_NAME}"'))

        # 6. Insert data into PostgreSQL
        print("Inserting data into PostgreSQL...")
        insert_query = text(f'INSERT INTO "{TABLE_NAME}" ({", ".join(quoted_column_names)}) VALUES ({", ".join([":" + col for col in column_names])})')

        # Execute the insert for each row
        for row in rows:
            row_dict = dict(zip(column_names, row))
            pg_conn.execute(insert_query, row_dict)
        
        # Commit the transaction to make the changes permanent
        pg_conn.commit()

        print("\n--- MIGRATION COMPLETE ---")
        print(f"Successfully migrated {len(rows)} rows to your Neon database.")
        print("You are now ready to deploy your application on Render.")

    except Exception as e:
        print(f"\n--- AN ERROR OCCURRED ---")
        print(f"Error: {e}")
    finally:
        # 7. Close all connections
        print("Closing database connections.")
        if 'sqlite_conn' in locals() and sqlite_conn:
            sqlite_conn.close()
        if 'pg_conn' in locals() and pg_conn:
            pg_conn.close()

if __name__ == '__main__':
    migrate_data()
