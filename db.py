import sqlite3
import os

# create ./db directory if it doesn't exist
if not os.path.exists('./db'):
    os.makedirs('./db')

conn = sqlite3.connect('./db/mydatabase.db')

def delete_prompt_by_id(conn, prompt_id):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prompts WHERE id = ?', (prompt_id,))
    conn.commit()

def delete_hash_if_no_prompts(conn, hash_value):
    cursor = conn.cursor()
    
    # Check if there are any prompts with the same hash value
    cursor.execute('SELECT COUNT(*) FROM prompts WHERE hash_value = ?', (hash_value,))
    count = cursor.fetchone()[0]

    # If there are no remaining prompts with the same hash, delete the hash
    if count == 0:
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
def create_prompt(conn, hash_value, prompt, negative, seed, height, width, timestamp, image_hash=None):
    cursor = conn.cursor()
    
    # Create or get the hash_value and the creation status flag
    hash_value, fetched = create_or_get_hash_id(conn, hash_value)

    # Insert the prompt with the associated hash_value
    cursor.execute('''
        INSERT INTO prompts (hash_value, prompt, negative, seed, height, width, timestamp, image_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (hash_value, prompt, negative, seed, height, width, timestamp, image_hash))
    
    conn.commit()

    return fetched