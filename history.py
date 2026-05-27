import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            data_json TEXT NOT NULL,
            row_count INTEGER DEFAULT 0,
            col_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_record(image_path: str, data: list) -> int:
    conn = get_connection()
    data_json = json.dumps(data, ensure_ascii=False)
    row_count = len(data)
    col_count = max((len(row) for row in data), default=0)
    cursor = conn.execute(
        "INSERT INTO records (image_path, data_json, row_count, col_count) VALUES (?, ?, ?, ?)",
        (image_path, data_json, row_count, col_count)
    )
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_all_records() -> list:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, image_path, row_count, col_count, created_at FROM records ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_record(record_id: int) -> dict:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM records WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    if row:
        result = dict(row)
        result['data'] = json.loads(result['data_json'])
        return result
    return None


def delete_record(record_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


init_db()
