"""Database connection manager."""

import os
from contextlib import contextmanager
from dotenv import load_dotenv
load_dotenv()

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


def _use_csv_fallback():
    return os.getenv("USE_CSV_FALLBACK", "false").lower() == "true"


def _dsn():
    """Supabase (prod) if SUPABASE_DB_URL is set, else local Docker (dev)."""
    url = os.getenv("SUPABASE_DB_URL")
    if url:
        return url
    return (
        f"host={os.getenv('DB_HOST', 'localhost')} "
        f"port={os.getenv('DB_PORT', '5438')} "
        f"dbname={os.getenv('DB_NAME', 'cx_monitor')} "
        f"user={os.getenv('DB_USER', 'cx_admin')} "
        f"password={os.getenv('DB_PASSWORD', 'cx_secret')}"
    )


@contextmanager
def get_conn():
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 not installed. Set USE_CSV_FALLBACK=true for Streamlit Cloud.")
    conn = psycopg2.connect(_dsn())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
