"""Offline unit test for osm_scan.py's element_to_record() — uses real,
representative Overpass JSON element shapes (a node and a way with
center, captured from Overpass's documented output format), not a live
API call (this sandbox can't reach overpass-api.de). Run directly:
python3 test_osm_scan_offline.py
"""
import sys
from osm_scan import element_to_record, build_query

# A real Overpass node element shape (shop) with a representative tag set.
NODE_ELEMENT = {
    "type": "node",
    "id": 123456789,
    "lat": 30.2711,
    "lon": -97.7437,
    "tags": {
        "shop": "coffee",
        "name": "Radio Coffee & Beer",
        "addr:housenumber": "4204",
        "addr:street": "Menchaca Rd",
        "addr:city": "Austin",
        "addr:state": "TX",
        "addr:postcode": "78704",
        "phone": "+1-512-220-2126",
        "website": "https://radiocoffeeandbeer.com",
        "contact:instagram": "radiocoffeeandbeer",
        "brand": "",
    },
}

# A real Overpass way element shape (amenity) with `out center` applied —
# ways don't have a direct lat/lon, only a computed "center" object.
WAY_ELEMENT = {
    "type": "way",
    "id": 987654321,
    "center": {"lat": 30.2850, "lon": -97.7341},
    "tags": {
        "amenity": "restaurant",
        "name": "Example Diner",
        "addr:housenumber": "100",
        "addr:street": "Main St",
    },
}

# An element with no name tag — must be filtered out (same logic that
# filters OSM infrastructure like benches/waste baskets that also carry
# amenity= tags).
NO_NAME_ELEMENT = {
    "type": "node",
    "id": 111,
    "lat": 1.0,
    "lon": 1.0,
    "tags": {"amenity": "bench"},
}

# A way with no center at all (malformed/truncated Overpass response) —
# must be filtered out, not crash.
NO_CENTER_WAY = {
    "type": "way",
    "id": 222,
    "tags": {"shop": "bakery", "name": "No Coordinates Bakery"},
}


def test_node():
    r = element_to_record(NODE_ELEMENT)
    assert r is not None
    assert r["id"] == "osm_node_123456789"
    assert r["name"] == "Radio Coffee & Beer"
    assert r["category"] == "coffee"
    assert r["address"] == "4204 Menchaca Rd"
    assert r["city"] == "Austin"
    assert r["region"] == "TX"
    assert r["country"] == "US"
    assert r["postcode"] == "78704"
    assert r["phone"] == "+1-512-220-2126"
    assert r["website"] == "https://radiocoffeeandbeer.com"
    assert r["social"] == "radiocoffeeandbeer"
    assert r["brand"] == ""  # empty string tag preserved, not silently dropped
    assert r["lat"] == 30.2711 and r["lon"] == -97.7437
    assert r["source"] == "osm"
    print("PASS: node element maps correctly")


def test_way_with_center():
    r = element_to_record(WAY_ELEMENT)
    assert r is not None
    assert r["id"] == "osm_way_987654321"
    assert r["category"] == "restaurant"
    assert r["lat"] == 30.2850 and r["lon"] == -97.7341
    print("PASS: way element with center maps correctly")


def test_no_name_filtered():
    assert element_to_record(NO_NAME_ELEMENT) is None
    print("PASS: unnamed element (e.g. a bench) correctly filtered out")


def test_no_center_filtered():
    assert element_to_record(NO_CENTER_WAY) is None
    print("PASS: way with no center/coordinates correctly filtered out, not crashed")


def test_query_bbox_order():
    # Overpass bbox order is (south, west, north, east) = (y0, x0, y1, x1)
    # — a swapped order here would silently query the wrong part of the
    # planet without ever raising an error, so this is worth pinning down.
    q = build_query(-97.8, 30.2, -97.7, 30.3)
    assert "(30.2,-97.8,30.3,-97.7)" in q, q
    print("PASS: Overpass bbox argument order is south,west,north,east as documented")


if __name__ == "__main__":
    test_node()
    test_way_with_center()
    test_no_name_filtered()
    test_no_center_filtered()
    test_query_bbox_order()
    print("\nALL OSM_SCAN OFFLINE TESTS PASSED")
