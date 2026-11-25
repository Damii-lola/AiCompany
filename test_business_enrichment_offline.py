"""Real test — spins up an actual local HTTP server serving real HTML
(with real schema.org JSON-LD, and a second page with no structured data
at all so the regex fallback path gets exercised too), then runs
service_business_enrichment against it over a real socket. Not a mock:
requests.get() really hits 127.0.0.1 and really parses what comes back.

Run directly: python3 test_business_enrichment_offline.py
"""
import threading
import http.server
import socketserver
import time

from service_business_enrichment import (
    enrich_businesses, enrichment_summary, _url_is_safe_to_fetch, _normalize_url,
)

SCHEMA_PAGE = b"""<!doctype html><html><head>
<title>Radio Coffee &amp; Beer - Austin, TX</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "CafeOrCoffeeShop",
  "name": "Radio Coffee & Beer",
  "telephone": "+1-512-220-2126",
  "priceRange": "$$",
  "aggregateRating": {"@type": "AggregateRating", "ratingValue": "4.6", "reviewCount": "812"},
  "openingHoursSpecification": [
    {"@type": "OpeningHoursSpecification", "dayOfWeek": ["Monday","Tuesday","Wednesday"], "opens": "07:00", "closes": "22:00"},
    {"@type": "OpeningHoursSpecification", "dayOfWeek": "Sunday", "opens": "08:00", "closes": "20:00"}
  ],
  "sameAs": ["https://www.facebook.com/radiocoffeeandbeer", "https://www.instagram.com/radiocoffeeandbeer/"]
}
</script>
</head><body><h1>Radio Coffee & Beer</h1></body></html>"""

NO_SCHEMA_PAGE = b"""<!doctype html><html><head><title>Joe's Diner</title></head>
<body>
<p>Call us at (512) 555-0199</p>
<a href="https://www.facebook.com/joesdiner">Facebook</a>
</body></html>"""

NOT_FOUND_PAGE = b"<html><body>gone</body></html>"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/schema":
            body = SCHEMA_PAGE
            code = 200
        elif self.path == "/noschema":
            body = NO_SCHEMA_PAGE
            code = 200
        else:
            body = NOT_FOUND_PAGE
            code = 404
        self.send_response(code)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep test output clean


def test_ssrf_guard():
    assert _url_is_safe_to_fetch("http://127.0.0.1:9/x") is False
    assert _url_is_safe_to_fetch("http://10.0.0.5/x") is False
    assert _url_is_safe_to_fetch("not a url") is False
    print("PASS: SSRF guard blocks loopback/private addresses and malformed URLs")


def test_normalize_url():
    assert _normalize_url("example.com") == "https://example.com"
    assert _normalize_url("http://example.com") == "http://example.com"
    assert _normalize_url("") == ""
    assert _normalize_url(None) == ""
    print("PASS: website URL normalization")


def test_live_enrichment():
    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            businesses = [
                {"id": "1", "name": "Radio Coffee & Beer", "website": f"http://127.0.0.1:{port}/schema"},
                {"id": "2", "name": "Joe's Diner", "website": f"http://127.0.0.1:{port}/noschema"},
                {"id": "3", "name": "Gone Business", "website": f"http://127.0.0.1:{port}/missing"},
                {"id": "4", "name": "No Website Listed", "website": None},
            ]

            # _url_is_safe_to_fetch blocks 127.0.0.1 by design (real SSRF
            # hardening) — temporarily allow loopback just for this one
            # local test server, the same way you'd use a test-only flag.
            import service_business_enrichment as mod
            real_guard = mod._url_is_safe_to_fetch
            mod._url_is_safe_to_fetch = lambda url: "127.0.0.1" in url
            try:
                out = enrich_businesses(businesses, max_candidates=10, budget_seconds=20, max_workers=4)
            finally:
                mod._url_is_safe_to_fetch = real_guard

            by_id = {b["id"]: b["enrichment"] for b in out}

            e1 = by_id["1"]
            assert e1["website_status"] == "reachable", e1
            assert e1["source_detail"] == "schema.org", e1
            assert e1["rating_value"] == 4.6, e1
            assert e1["review_count"] == 812, e1
            assert e1["price_range"] == "$$", e1
            assert e1["phone_confirmed"] == "+1-512-220-2126", e1
            assert "facebook" in e1["social_links"] and e1["social_links"]["facebook"] == "radiocoffeeandbeer", e1
            assert "instagram" in e1["social_links"], e1
            assert "Monday/Tuesday/Wednesday 07:00-22:00" in e1["hours_text"], e1
            assert e1["mobile_friendly"] is True, e1
            print("PASS: schema.org JSON-LD page fully parsed (rating, hours, phone, price, social)")

            e2 = by_id["2"]
            assert e2["website_status"] == "reachable", e2
            assert e2["source_detail"] == "regex_fallback", e2
            assert e2["phone_confirmed"] and "555" in e2["phone_confirmed"], e2
            assert e2["social_links"].get("facebook") == "joesdiner", e2
            assert e2["rating_value"] is None  # never invented when absent
            print("PASS: no-schema page falls back to regex phone/social extraction, no fabricated rating")

            e3 = by_id["3"]
            assert e3["website_status"] == "unreachable", e3
            assert "404" in (e3["error"] or ""), e3
            print("PASS: 404 page reported as unreachable with real HTTP status in error, not silently skipped")

            e4 = by_id["4"]
            assert e4["website_status"] == "no_website", e4
            print("PASS: business with no website gets no_website status, no fetch attempted")

            summary = enrichment_summary(out)
            assert summary["websites_checked"] == 3
            assert summary["websites_reachable"] == 2
            assert summary["avg_rating_where_available"] == 4.6
            assert summary["rating_sample_size"] == 1
            print("PASS: enrichment_summary computes real aggregates (checked/reachable/avg rating)")
        finally:
            httpd.shutdown()


def test_candidate_cap():
    businesses = [{"id": str(i), "name": f"Biz {i}", "website": "http://127.0.0.1:1/x"} for i in range(5)]
    out = enrich_businesses(businesses, max_candidates=2, budget_seconds=5, max_workers=2)
    checked = [b for b in out if b["enrichment"]["website_status"] != "not_checked"]
    not_checked = [b for b in out if b["enrichment"]["website_status"] == "not_checked"]
    assert len(not_checked) == 3, [b["enrichment"] for b in out]
    assert len(checked) == 2
    print("PASS: per-report candidate cap leaves the rest honestly marked not_checked, never dropped from the report")


if __name__ == "__main__":
    test_ssrf_guard()
    test_normalize_url()
    test_live_enrichment()
    test_candidate_cap()
    print("\nALL BUSINESS ENRICHMENT OFFLINE TESTS PASSED")
