"""Database connection manager."""

import os
from contextlib import contextmanager

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


def _use_csv_fallback():
    return os.getenv("USE_CSV_FALLBACK", "false").lower() == "true"


@contextmanager
def get_conn():
    """Context manager for PostgreSQL connection.

    Usage:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")
    """
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 not installed. Set USE_CSV_FALLBACK=true for Streamlit Cloud.")

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5435")),
        dbname=os.getenv("DB_NAME", "cx_monitor"),
        user=os.getenv("DB_USER", "cx_admin"),
        password=os.getenv("DB_PASSWORD", "cx_secret"),
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
