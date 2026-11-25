"""Real end-to-end smoke test of service_sba_trade_area_snapshot.fulfil()
with enrichment wired in — monkeypatches only the render-service network
call (_fetch_trade_area, since this sandbox can't reach the live Render
deployment), everything downstream (pandas filtering, real enrichment HTTP
calls to a real local server, CSV/XLSX/PDF writing, manifest) runs for
real. Run directly: python3 test_sba_snapshot_e2e_offline.py
"""
import os
import json
import shutil
import threading
import http.server
import socketserver

import service_sba_trade_area_snapshot as sba
import service_business_enrichment as enrich_mod

SCHEMA_PAGE = b"""<!doctype html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"Dentist","name":"Test Dental",
 "telephone":"+1-614-555-0100","priceRange":"$$$",
 "aggregateRating":{"ratingValue":"4.8","reviewCount":"230"},
 "openingHoursSpecification":[{"dayOfWeek":["Monday","Tuesday"],"opens":"08:00","closes":"17:00"}]}
</script></head><body>Test Dental</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(SCHEMA_PAGE)

    def log_message(self, *a):
        pass


def main():
    out_dir = "./test_trade_area_e2e"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            real_guard = enrich_mod._url_is_safe_to_fetch
            enrich_mod._url_is_safe_to_fetch = lambda url: "127.0.0.1" in url

            fake_result = {
                "summary": {"competitor_count": 2, "no_website_count": 1, "nearest_miles": 0.4},
                "matched_categories": ["dental clinic"],
                "center": {"lat": 39.9612, "lon": -82.9988},
                "businesses": [
                    {
                        "id": "b1", "name": "Test Dental", "category": "dental clinic",
                        "category_extra": None, "address": "1 Main St", "city": "Columbus",
                        "region": "OH", "country": "US", "postcode": "43215",
                        "phone": "614-555-0100", "website": f"http://127.0.0.1:{port}/",
                        "email": None, "social": None, "brand": None, "confidence": 0.9,
                        "operating_status": "open", "lon": -82.9988, "lat": 39.9615,
                        "distance_miles": 0.4,
                    },
                    {
                        "id": "b2", "name": "No Site Dental", "category": "dental clinic",
                        "category_extra": None, "address": "2 Main St", "city": "Columbus",
                        "region": "OH", "country": "US", "postcode": "43215",
                        "phone": None, "website": None, "email": None, "social": None,
                        "brand": None, "confidence": 0.7, "operating_status": "open",
                        "lon": -82.999, "lat": 39.962, "distance_miles": 0.6,
                    },
                ],
            }
            sba._fetch_trade_area = lambda spec: fake_result

            spec = {
                "address": "Columbus, OH", "category": "dental clinic", "radius_miles": 5,
                "country": "US", "output_formats": ["csv", "xlsx", "pdf"],
            }
            files = sba.fulfil(spec, out_dir)
            assert len(files) == 4, files  # csv, xlsx, pdf, manifest.json
            for f in files:
                assert os.path.getsize(f) > 0, f"{f} is empty"

            manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
            assert manifest["competitor_count"] == 2
            assert manifest["enrichment"]["websites_checked"] == 1
            assert manifest["enrichment"]["websites_reachable"] == 1
            assert manifest["enrichment"]["avg_rating_where_available"] == 4.8
            print("PASS: fulfil() end-to-end — CSV/XLSX/PDF/manifest all written, "
                  "real enrichment data flowed through into the manifest")

            import pandas as pd
            df = pd.read_csv(os.path.join(out_dir, [f for f in files if f.endswith(".csv")][0].split("/")[-1]) if False else
                              [f for f in files if f.endswith(".csv")][0])
            assert "enrich_rating_value" in df.columns
            assert float(df.loc[df["name"] == "Test Dental", "enrich_rating_value"].iloc[0]) == 4.8
            print("PASS: enrichment columns present in exported CSV with real values")
        finally:
            enrich_mod._url_is_safe_to_fetch = real_guard
            httpd.shutdown()

    shutil.rmtree(out_dir, ignore_errors=True)
    print("\nALL SBA SNAPSHOT E2E OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
