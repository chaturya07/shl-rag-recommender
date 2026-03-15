"""
SQLite database layer for search history and analytics.
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "shl_recommender.db"


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                result_count INTEGER,
                method TEXT,
                query_time_ms REAL,
                assessment_type TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS query_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                result_count INTEGER,
                method TEXT,
                query_time_ms REAL,
                cached INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_session ON search_history(session_id);
            CREATE INDEX IF NOT EXISTS idx_created ON search_history(created_at);
        """)


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_search(session_id: str, query: str, result_count: int,
                method: str, query_time_ms: float, assessment_type: str = None):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO search_history 
            (session_id, query, result_count, method, query_time_ms, assessment_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, query, result_count, method, query_time_ms, assessment_type))

        conn.execute("""
            INSERT INTO query_analytics
            (query, result_count, method, query_time_ms)
            VALUES (?, ?, ?, ?)
        """, (query, result_count, method, query_time_ms))


def get_history(session_id: str, limit: int = 20):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT query, result_count, method, assessment_type, created_at
            FROM search_history
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()
        return [dict(r) for r in rows]


def clear_history(session_id: str):
    with get_conn() as conn:
        conn.execute("DELETE FROM search_history WHERE session_id = ?", (session_id,))


def get_analytics_summary():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM query_analytics").fetchone()[0]
        avg_time = conn.execute(
            "SELECT AVG(query_time_ms) FROM query_analytics"
        ).fetchone()[0] or 0

        top_queries = conn.execute("""
            SELECT query, COUNT(*) as count
            FROM query_analytics
            GROUP BY LOWER(query)
            ORDER BY count DESC
            LIMIT 10
        """).fetchall()

        return {
            "total_queries": total,
            "avg_query_time_ms": round(avg_time, 2),
            "top_queries": [dict(r) for r in top_queries]
        }


# Initialize on import
init_db()
