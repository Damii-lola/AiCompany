"""Real test — a local HTTP server stands in for a Socrata state portal,
serving real Socrata-shaped JSON array-of-row-dict responses (one with
Colorado's actually-confirmed field names, one with unknown/generic field
names to exercise the heuristic-field-detection fallback path). Run
directly: python3 test_business_registry_offline.py
"""
import json
import threading
import http.server
import socketserver

import service_business_registry as reg
from service_business_registry import (
    enrich_with_registry, registry_summary, _normalize_name, _name_matches,
)

CO_ROWS = [
    {"entityid": "1001", "entityname": "JOES PIZZA LLC", "entitystatus": "Good Standing",
     "principalcity": "Denver", "principalstate": "CO", "principalzipcode": "80202"},
    {"entityid": "1002", "entityname": "UNRELATED BUSINESS INC", "entitystatus": "Delinquent",
     "principalcity": "Boulder", "principalstate": "CO", "principalzipcode": "80301"},
]

# Simulates a state whose exact field names were never confirmed — only
# generic "name"/"status"/"city"-ish keys, to prove the heuristic fallback
# (not the hardcoded Colorado field names) finds them.
GENERIC_ROWS = [
    {"current_entity_name": "Example Diner", "entity_status_description": "Active",
     "business_city": "Columbus"},
]


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if "generic" in self.path:
            body = json.dumps(GENERIC_ROWS).encode()
        elif "empty" in self.path:
            body = b"[]"
        else:
            body = json.dumps(CO_ROWS).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_normalize_and_match():
    assert _normalize_name("Joe's Pizza, LLC") == _normalize_name("JOES PIZZA LLC")
    assert _name_matches("Joe's Pizza LLC", "JOES PIZZA LLC")
    assert not _name_matches("Joe's Pizza LLC", "Totally Different Co")
    print("PASS: name normalization strips punctuation/legal suffixes/case for fuzzy comparison")


def test_uncovered_state_short_circuits():
    out = enrich_with_registry([{"name": "Some Shop", "region": "ZZ"}])
    r = out[0]["registry"]
    assert r["covered"] is False
    assert "ZZ" in r["error"]
    print("PASS: a region with no wired-in state registry is reported honestly, no HTTP call made")


def test_live_lookup_confirmed_fields_and_fallback():
    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            # Point CO's real config at the local test server instead of
            # data.colorado.gov for this test — same contract, no live
            # network call (this sandbox can't reach the real Socrata
            # hosts anyway).
            real_sources = reg.STATE_SOCRATA_SOURCES
            reg.STATE_SOCRATA_SOURCES = {
                "CO": {**real_sources["CO"], "domain": f"127.0.0.1:{port}/co", "scheme": "http"},
                "XX": {"domain": f"127.0.0.1:{port}/generic", "resource_id": "r1",
                       "domain_confirmed": True, "scheme": "http"},
            }
            try:
                businesses = [
                    {"name": "Joe's Pizza LLC", "region": "CO"},
                    {"name": "Totally Unmatched Business", "region": "CO"},
                    {"name": "Example Diner", "region": "XX"},
                ]
                out = enrich_with_registry(businesses, max_candidates=10, budget_seconds=10, max_workers=3)

                r0 = out[0]["registry"]
                assert r0["covered"] is True
                assert r0["matched"] is True, r0
                assert r0["status_text"] == "Good Standing", r0
                assert r0["city"] == "Denver", r0
                print("PASS: Colorado's confirmed exact field names (entityname/entitystatus/"
                      "principalcity) read correctly, with a real fuzzy name match")

                r1 = out[1]["registry"]
                assert r1["matched"] is False, r1
                print("PASS: a business name with no real match in the returned rows is NOT "
                      "falsely reported as found (full-text search alone isn't trusted)")

                r2 = out[2]["registry"]
                assert r2["matched"] is True, r2
                assert r2["status_text"] == "Active", r2
                assert r2["registry_name"] == "Example Diner", r2
                print("PASS: generic/unconfirmed field names are still found via the "
                      "name/status/city heuristic substring match, not hardcoded guesses")

                summary = registry_summary(out)
                assert summary["registry_checks_attempted"] == 3
                assert summary["states_with_registry_coverage"] == 3
                assert summary["confirmed_registered_entities"] == 2
                print("PASS: registry_summary computes real aggregate counts")
            finally:
                reg.STATE_SOCRATA_SOURCES = real_sources
        finally:
            httpd.shutdown()


if __name__ == "__main__":
    test_normalize_and_match()
    test_uncovered_state_short_circuits()
    test_live_lookup_confirmed_fields_and_fallback()
    print("\nALL BUSINESS REGISTRY OFFLINE TESTS PASSED")
