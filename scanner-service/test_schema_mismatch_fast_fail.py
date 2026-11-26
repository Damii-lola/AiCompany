"""Real regression test for a live incident (Nov 2026): a schema change
(businesses.basic_category/category_hierarchy) shipped before all live
TiDB shards had the matching ALTER TABLE applied, and scan_region()'s
split-and-retry logic — correctly designed for TRANSIENT per-region
failures like an OOM on a dense region, where a smaller bbox genuinely
has a better chance — instead spent an entire ~5.5h scheduled run
quartering every single tile down to its smallest split size and failing
identically at every depth, because a schema mismatch fails the exact
same way no matter how small the region is. Caught live via the GitHub
Actions run log, the run was cancelled, and scan_region() was fixed to
detect this error class (MySQL/TiDB errno 1054/1146/1049) and abort the
whole run immediately with one clear message instead.

Requires a real local MySQL/MariaDB at the DATABASE_URL_1 env var (same
convention as test_osm_scan_offline.py). Run directly:
DATABASE_URL_1="mysql://chat_test:chat_test@localhost:3306/chat_test_tidb?ssl=false" \
    python3 test_schema_mismatch_fast_fail.py
"""
import os
import sys

import pandas as pd

import scan_worker as sw


def _get_pg():
    url = os.environ.get("DATABASE_URL_1")
    if not url:
        print("Set DATABASE_URL_1 to a real local MySQL/MariaDB test database to run this.")
        sys.exit(1)
    return sw.pg_connect(url)


class _FakeResult:
    def __init__(self, df):
        self._df = df

    def fetchdf(self):
        return self._df


def test_real_schema_mismatch_fast_fails_instead_of_splitting():
    pg = _get_pg()
    lookups = sw.Lookups(pg)

    with pg.cursor() as cur:
        # Reproduce the EXACT live incident: drop the two columns this
        # test DB otherwise has (from database/schema.mysql.sql), insert
        # through the real pg_upsert_businesses/scan_region path, and
        # confirm it fails FAST with a clear RuntimeError rather than
        # recursing into the quadrant-splitting retry loop.
        cur.execute("ALTER TABLE businesses DROP COLUMN IF EXISTS basic_category")
        cur.execute("ALTER TABLE businesses DROP COLUMN IF EXISTS category_hierarchy")
    pg.commit()

    try:
        record = {
            "id": "schema_mismatch_regression_test", "name": "Test Biz", "category": "coffee_shop",
            "category_extra": None, "address": "1 St", "city": "X", "region": "OH", "country": "US",
            "postcode": "1", "phone": None, "website": None, "email": None, "social": None,
            "brand": None, "confidence": 0.9, "operating_status": "open", "lon": 1.0, "lat": 1.0,
            "source": "overture", "basic_category": "coffee_shop", "category_hierarchy": "[]",
        }

        class FakeCon:
            def execute(self, sql):
                return _FakeResult(pd.DataFrame([record]))

        select = ",".join(record.keys())
        cols = set(select.split(","))

        try:
            sw.scan_region(FakeCon(), pg, lookups, select, cols, -1, -1, 1, 1, depth=0)
            raise AssertionError("scan_region did not raise — the schema mismatch went undetected")
        except RuntimeError as e:
            assert "Schema mismatch" in str(e), e
            assert "splitting into smaller regions cannot fix" in str(e), e
            print("PASS: a real schema mismatch (dropped columns, real MySQL errno 1054) "
                  "fails scan_region() FAST with one clear message, not a wasted multi-hour "
                  "quadrant-splitting retry loop")

        # Also confirm _is_schema_mismatch_error itself, directly, against
        # the real exception pg_upsert_businesses raises.
        try:
            sw.pg_upsert_businesses(pg, lookups, [record])
            raise AssertionError("pg_upsert_businesses did not raise against the mismatched schema")
        except Exception as e:
            assert sw._is_schema_mismatch_error(e), (type(e), e)
            print("PASS: _is_schema_mismatch_error correctly classifies the real raised exception")
    finally:
        # Restore the columns so this test is repeatable and doesn't
        # leave the local test database in a broken state for any other
        # test file that runs after it.
        with pg.cursor() as cur:
            cur.execute("ALTER TABLE businesses ADD COLUMN IF NOT EXISTS basic_category varchar(128)")
            cur.execute("ALTER TABLE businesses ADD COLUMN IF NOT EXISTS category_hierarchy text")
        pg.commit()


def test_genuine_transient_error_still_splits_normally():
    """The fast-fail path must only catch real schema-mismatch errnos —
    a genuine transient failure (timeout, OOM, connection hiccup) has to
    keep using the existing split-and-retry behavior unchanged."""
    class FailingResult:
        def fetchdf(self):
            raise MemoryError("simulated OOM on a dense region")

    class FakeCon:
        def execute(self, sql):
            return FailingResult()

    orig_depth, orig_min = sw.MAX_SPLIT_DEPTH, sw.MIN_SPLIT_DEG
    sw.MAX_SPLIT_DEPTH = 1
    sw.MIN_SPLIT_DEG = 10
    try:
        try:
            sw.scan_region(FakeCon(), None, None, "id", set(), -10, -10, 10, 10, depth=0)
            raise AssertionError("expected MemoryError to propagate")
        except RuntimeError:
            raise AssertionError("a genuine transient MemoryError was wrongly classified "
                                  "as a schema mismatch")
        except MemoryError:
            print("PASS: a genuine transient error (MemoryError) still propagates as itself "
                  "after exhausting the normal split-retry path, unaffected by the new check")
    finally:
        sw.MAX_SPLIT_DEPTH, sw.MIN_SPLIT_DEG = orig_depth, orig_min


if __name__ == "__main__":
    test_real_schema_mismatch_fast_fails_instead_of_splitting()
    test_genuine_transient_error_still_splits_normally()
    print("\nALL SCHEMA MISMATCH FAST-FAIL REGRESSION TESTS PASSED")
