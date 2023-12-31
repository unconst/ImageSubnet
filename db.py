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
        batch_id INTEGER,
        hash_value TEXT,
        image_order_id INTEGER,
        uid INTEGER,
        prompt TEXT,
        negative TEXT,
        seed INTEGER,
        height INTEGER,
        width INTEGER,
        timestamp INTEGER,
        image_hash TEXT,
        FOREIGN KEY (hash_value) REFERENCES hashes (hash_value),
        FOREIGN KEY (image_hash) REFERENCES hashes (hash_value)
    )
''')

# Create the 'prompts' table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompts (
        id INTEGER PRIMARY KEY,
        batch_id INTEGER,
        hash_value TEXT,
        image_order_id INTEGER,
        uid INTEGER,
        prompt TEXT,
        negative TEXT,
        seed INTEGER,
        height INTEGER,
        width INTEGER,
        timestamp INTEGER,
        FOREIGN KEY (hash_value) REFERENCES hashes (hash_value)
    )
''')

# Create the 'batches' table if it doesn't exist
cursor.execute('''
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY,
        timestamp INTEGER
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

def delete_batch_if_no_prompts(conn, batch_id):
    cursor = conn.cursor()
    
    # Check if there are any prompts with the same batch_id
    cursor.execute('SELECT COUNT(*) FROM prompts WHERE batch_id = ?', (batch_id,))
    count = cursor.fetchone()[0]

    # Check if there are any i2iprompts with the same batch_id
    cursor.execute('SELECT COUNT(*) FROM i2iprompts WHERE batch_id = ?', (batch_id,))
    i2icount = cursor.fetchone()[0]

    # If there are no remaining prompts with the same hash, delete the hash
    if count == 0 and i2icount == 0:
        cursor.execute('DELETE FROM batches WHERE id = ?', (batch_id,))
        conn.commit()

def delete_prompts_by_timestamp(conn, timestamp):
    cursor = conn.cursor()
    
    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value, batch_id FROM prompts WHERE timestamp > ?', (timestamp,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value, batch_id in prompts_to_delete:
        # Delete the prompt
        delete_prompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

        delete_batch_if_no_prompts(conn, batch_id)

    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value FROM i2iprompts WHERE timestamp > ?', (timestamp,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value in prompts_to_delete:
        # Delete the prompt
        delete_i2iprompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

        delete_batch_if_no_prompts(conn, batch_id)


def delete_prompts_by_uid(conn, uid):
    cursor = conn.cursor()
    
    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value, batch_id FROM prompts WHERE uid = ?', (uid,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value, batch_id in prompts_to_delete:
        # Delete the prompt
        delete_prompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

        delete_batch_if_no_prompts(conn, batch_id)

    # Get all prompts with timestamps beyond the specified value
    cursor.execute('SELECT id, hash_value FROM i2iprompts WHERE uid = ?', (uid,))
    prompts_to_delete = cursor.fetchall()

    for prompt_id, hash_value in prompts_to_delete:
        # Delete the prompt
        delete_i2iprompt_by_id(conn, prompt_id)

        # Check and delete the associated hash if necessary
        delete_hash_if_no_prompts(conn, hash_value)

        delete_batch_if_no_prompts(conn, batch_id)

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
def create_prompt(conn, batch_id, hash_value, image_order_id, uid, prompt, negative, seed, height, width, timestamp, input_image_hash=None):
    cursor = conn.cursor()
    
    # Create or get the hash_value and the creation status flag
    hash_value, fetched = create_or_get_hash_id(conn, hash_value)

    if input_image_hash is None:
        # Insert the prompt with the associated hash_value
        cursor.execute('''
            INSERT INTO prompts (batch_id, hash_value, image_order_id, uid, prompt, negative, seed, height, width, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (batch_id, hash_value, image_order_id, uid, prompt, negative, seed, height, width, timestamp))
    else:
        # Insert the prompt with the associated hash_value and input_image_hash
        cursor.execute('''
            INSERT INTO i2iprompts (batch_id, hash_value, image_order_id, uid, prompt, negative, seed, height, width, image_hash, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (batch_id, hash_value, image_order_id, uid, prompt, negative, seed, height, width, input_image_hash, timestamp))
    
    conn.commit()

    return fetched

def create_batch(conn, timestamp):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO batches (timestamp) VALUES (?)', (int(timestamp),))
    conn.commit()
    return cursor.lastrowid

def get_batch(conn, batch_id):
    # return all the prompts that are tied to the batch_id
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM prompts WHERE batch_id = ?', (batch_id,))
    prompts = cursor.fetchall()

    # create a list of Prompt classes
    return [Prompt(prompt) for prompt in prompts]

# get random batch id within timestamp range
def get_random_batch_id(conn, start_timestamp = None, end_timestamp = None):
    cursor = conn.cursor()

    if start_timestamp is None:
        start_timestamp = 0
    if end_timestamp is None:
        end_timestamp = 9999999999

    cursor.execute('''
        SELECT id FROM batches
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY RANDOM() LIMIT 1
    ''', (start_timestamp, end_timestamp))
    batch_id = cursor.fetchone()

    if batch_id is None:
        return None
    else:
        return batch_id[0]
    
# get random batch and all the associated prompts
def get_prompts_of_random_batch(conn, start_timestamp = None, end_timestamp = None):
    batch_id = get_random_batch_id(conn, start_timestamp, end_timestamp)
    if batch_id is None:
        return None
    else:
        return get_batch(conn, batch_id)
    
# create a class object which takes in a prompt from sql and returns a prompt object
class Prompt:
    def __init__(self, prompt):
        self.id = prompt[0]
        self.batch_id = prompt[1]
        self.hash_value = prompt[2]
        self.image_order_id = prompt[3]
        self.uid = prompt[4]
        self.prompt = prompt[5]
        self.negative = prompt[6]
        self.seed = prompt[7]
        self.height = prompt[8]
        self.width = prompt[9]
        self.timestamp = prompt[10]
        if len(prompt) > 11:
            self.input_image_hash = prompt[11]
    
    def __str__(self):
        return str(self.__dict__)
    