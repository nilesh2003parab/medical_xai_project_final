import sqlite3


def init_db():
    conn = sqlite3.connect("patient_data.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT,
            name TEXT,
            prediction TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()