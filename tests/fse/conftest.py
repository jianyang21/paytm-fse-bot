import os
import uuid
from pathlib import Path

import fakeredis
import psycopg
import pytest
from psycopg import sql

from teammates.config import read_env_file
from teammates.fse.store import Store

PHONE = "919999900000"
DSN = os.environ.get("TEST_DATABASE_URL") or read_env_file(Path(".env")).get("TEST_DATABASE_URL")


@pytest.fixture
def store():
    """A Store on a throwaway Postgres schema (dropped afterwards) with a fake Redis."""
    if not DSN:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"t_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(DSN, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    s = Store(DSN, cache=fakeredis.FakeRedis(), schema=schema)
    try:
        s.migrate()
        s.seed_demo(PHONE)
        yield s
    finally:
        s.pool.close()
        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
