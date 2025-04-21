import sqlite3

def get_db_connection():
    conn = sqlite3.connect('fundusdb1.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_doctor_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            phone TEXT,
            medical_reg_no TEXT,
            hospital TEXT
        )
    ''')
    conn.commit()
    conn.close()

def create_patient_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            dob TEXT,
            age INTEGER,
            patient_id TEXT UNIQUE,
            contact TEXT,
            visit_date TEXT,
            medical_history TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_image_path_column():
    conn = get_db_connection()
    conn.execute("ALTER TABLE patients ADD COLUMN image_path TEXT;")
    conn.commit()
    conn.close()

def add_results_column():
    conn = get_db_connection()
    conn.execute("ALTER TABLE patients ADD COLUMN results TEXT;")
    conn.commit()
    conn.close()
    
def add_eye_side_column():
    try:
        conn = get_db_connection()
        conn.execute("ALTER TABLE patients ADD COLUMN eye_side TEXT;")
        conn.commit()
        conn.close()
    except sqlite3.OperationalError:
        pass  # Column already exists



