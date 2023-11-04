import sqlite3
import os

# create ./db directory if it doesn't exist
if not os.path.exists('./db'):
    os.makedirs('./db')

conn = sqlite3.connect('./db/imagenet.db')

# Create a cursor
cursor = conn.cursor()

# Create the 'hashes' table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS hashes (
        hash_value TEXT PRIMARY KEY
    )
''')

# Create the 'prompts' table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS i2iprompts (
        id INTEGER PRIMARY KEY,
        hash_value TEXT,
        prompt TEXT,
        negative TEXT,
        seed INTEGER,
        height INTEGER,
        width INTEGER,
        image_hash TEXT,
        timestamp INTEGER,
        FOREIGN KEY (hash_value) REFERENCES hashes (hash_value),
        FOREIGN KEY (image_hash) REFERENCES hashes (hash_value)
    )
''')

# Create the 'prompts' table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY,
        hash_value TEXT,
        prompt TEXT,
        negative TEXT,
        seed INTEGER,
        height INTEGER,
        width INTEGER,
        timestamp INTEGER,
        FOREIGN KEY (hash_value) REFERENCES hashes (hash_value)
    )
''')

cursor = None

def delete_prompt_by_id(conn, prompt_id):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prompts WHERE id = ?', (prompt_id,))
    conn.commit()

def delete_i2iprompt_by_id(conn, prompt_id):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM i2iprompts WHERE id = ?', (prompt_id,))
    conn.commit()


def delete_hash_if_no_prompts(conn, hash_value):
    cursor = conn.cursor()
    
    # Check if there are any prompts with the same hash value
    cursor.execute('SELECT COUNT(*) FROM prompts WHERE hash_value = ?', (hash_value,))
    count = cursor.fetchone()[0]

    # Check if there are any i2iprompts with the same hash value
    cursor.execute('SELECT COUNT(*) FROM i2iprompts WHERE hash_value = ?', (hash_value,))
    i2icount = cursor.fetchone()[0]

    # If there are no remaining prompts with the same hash, delete the hash
    if count == 0 and i2icount == 0:
        cursor.execute('DELETE FROM hashes WHERE hash_value = ?', (hash_value,))
        conn.commit()

def delete_prompts_by_timestamp(conn, timestamp):
    cursor = conn.cursor()
    
    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value FROM prompts WHERE timestamp > ?', (timestamp,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value in prompts_to_delete:
        # Delete the prompt
        delete_prompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value FROM i2iprompts WHERE timestamp > ?', (timestamp,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value in prompts_to_delete:
        # Delete the prompt
        delete_i2iprompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

def create_or_get_hash_id(conn, hash_value):
    cursor = conn.cursor()
    
    # Try to retrieve a row with the given hash_value
    cursor.execute('SELECT hash_value FROM hashes WHERE hash_value = ?', (hash_value,))
    existing_hash = cursor.fetchone()
    
    if existing_hash:
        # If the hash already exists, return the hash_value and a flag indicating it was fetched
        return existing_hash[0], True
    else:
        # If the hash doesn't exist, create it, return the hash_value, and a flag indicating it was created
        cursor.execute('INSERT INTO hashes (hash_value) VALUES (?)', (hash_value,))
        conn.commit()
        return hash_value, False


# returns a boolean value indicating whether the hash already existed (True) or was created (False)
def create_prompt(conn, hash_value, prompt, negative, seed, height, width, timestamp, input_image_hash=None):
    cursor = conn.cursor()
    
    # Create or get the hash_value and the creation status flag
    hash_value, fetched = create_or_get_hash_id(conn, hash_value)

    if input_image_hash is None:
        # Insert the prompt with the associated hash_value
        cursor.execute('''
            INSERT INTO prompts (hash_value, prompt, negative, seed, height, width, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (hash_value, prompt, negative, seed, height, width, timestamp))
    else:
        # Insert the prompt with the associated hash_value and input_image_hash
        cursor.execute('''
            INSERT INTO prompts (hash_value, prompt, negative, seed, height, width, image_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (hash_value, prompt, negative, seed, height, width, input_image_hash, timestamp))
    
    conn.commit()

    return fetched