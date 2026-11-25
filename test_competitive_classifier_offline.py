"""Real unit tests for service_competitive_classifier.py, using real
Overture taxonomy hierarchy shapes (confirmed via docs.overturemaps.org
and the OvertureMaps/schema repo during this session's research, not
invented). Run directly: python3 test_competitive_classifier_offline.py
"""
from service_competitive_classifier import (
    classify_pair, classify_businesses, classification_summary,
)

PIZZA_PLACE = {
    "category": "pizza_restaurant",
    "category_hierarchy": '["food_and_drink","restaurant","fast_food_restaurant","pizza_restaurant"]',
}
ANOTHER_PIZZA_PLACE = {
    "category": "pizza_restaurant",
    "category_hierarchy": '["food_and_drink","restaurant","fast_food_restaurant","pizza_restaurant"]',
}
SUB_SHOP = {
    # Same immediate parent (fast_food_restaurant) as pizza, different leaf.
    "category": "sandwich_shop",
    "category_hierarchy": '["food_and_drink","restaurant","fast_food_restaurant","sandwich_shop"]',
}
FINE_DINING = {
    # Same grandparent (restaurant), different branch below it.
    "category": "fine_dining_restaurant",
    "category_hierarchy": '["food_and_drink","restaurant","fine_dining_restaurant"]',
}
COFFEE_SHOP = {
    # Only the generic top-level ("food_and_drink") in common.
    "category": "coffee_shop",
    "category_hierarchy": '["food_and_drink","cafe","coffee_shop"]',
}
HARDWARE_STORE = {
    # No overlap at all.
    "category": "hardware_store",
    "category_hierarchy": '["retail","home_improvement_store","hardware_store"]',
}
NO_HIERARCHY_PIZZA = {
    # Same leaf as PIZZA_PLACE, but scanned before the hierarchy column
    # existed (realistic — most of the live 18.8M rows are like this).
    "category": "pizza_restaurant", "category_hierarchy": None,
}
NO_HIERARCHY_UNRELATED = {
    "category": "dentist", "category_extra": None, "category_hierarchy": None,
}
NO_HIERARCHY_WITH_EXTRA_OVERLAP = {
    "category": "fast_food_restaurant", "category_extra": "pizza_restaurant",
    "category_hierarchy": None,
}


def test_identical_leaf_is_direct():
    r = classify_pair(PIZZA_PLACE, ANOTHER_PIZZA_PLACE)
    assert r["classification"] == "direct" and r["basis"] == "hierarchy", r
    print("PASS: identical hierarchy path -> direct")


def test_same_immediate_parent_is_direct():
    r = classify_pair(PIZZA_PLACE, SUB_SHOP)
    assert r["classification"] == "direct" and r["basis"] == "hierarchy", r
    print("PASS: same immediate parent (both fast_food_restaurant), different leaf -> direct")


def test_same_grandparent_is_indirect():
    r = classify_pair(PIZZA_PLACE, FINE_DINING)
    assert r["classification"] == "indirect", r
    print("PASS: shared grandparent (restaurant) but different branch -> indirect")


def test_only_generic_top_level_is_related():
    r = classify_pair(PIZZA_PLACE, COFFEE_SHOP)
    assert r["classification"] == "related", r
    print("PASS: only the generic top-level category (food_and_drink) shared -> related")


def test_no_overlap_is_irrelevant():
    r = classify_pair(PIZZA_PLACE, HARDWARE_STORE)
    assert r["classification"] == "irrelevant", r
    print("PASS: no shared ancestor at all -> irrelevant")


def test_fallback_when_hierarchy_missing():
    r = classify_pair(NO_HIERARCHY_PIZZA, {"category": "pizza_restaurant"})
    assert r["classification"] == "direct" and r["basis"] == "leaf_category_fallback", r

    r2 = classify_pair(NO_HIERARCHY_PIZZA, NO_HIERARCHY_UNRELATED)
    assert r2["classification"] == "irrelevant" and r2["basis"] == "leaf_category_fallback", r2

    r3 = classify_pair({"category": "pizza_restaurant"}, NO_HIERARCHY_WITH_EXTRA_OVERLAP)
    assert r3["classification"] == "indirect" and r3["basis"] == "leaf_category_fallback", r3
    print("PASS: missing hierarchy data falls back honestly to leaf/alternate-category "
          "comparison, correctly labeled with basis='leaf_category_fallback'")


def test_one_sided_hierarchy_falls_back_too():
    # Only one side has hierarchy data — real scenario when comparing a
    # freshly-scanned business against an old pre-column one.
    r = classify_pair(PIZZA_PLACE, NO_HIERARCHY_PIZZA)
    assert r["basis"] == "leaf_category_fallback", r
    assert r["classification"] == "direct", r
    print("PASS: one-sided hierarchy data correctly falls back rather than half-comparing")


def test_batch_classify_and_summary():
    target_category = "pizza_restaurant"
    businesses = [dict(ANOTHER_PIZZA_PLACE), dict(SUB_SHOP), dict(FINE_DINING),
                  dict(COFFEE_SHOP), dict(HARDWARE_STORE)]
    out = classify_businesses(target_category, businesses)
    assert out[0]["competitive_classification"]["classification"] == "direct"
    summary = classification_summary(out)
    assert summary["direct"] == 2  # ANOTHER_PIZZA_PLACE + SUB_SHOP (same immediate parent)
    assert summary["indirect"] == 1
    assert summary["related"] == 1
    assert summary["irrelevant"] == 1
    assert summary["classified_from_real_hierarchy_data"] == 5
    print("PASS: classify_businesses + classification_summary over a real mixed batch")


def test_insufficient_data_never_crashes():
    r = classify_pair({"category": None}, {"category": "pizza_restaurant"})
    assert r["classification"] is None and r["basis"] == "insufficient_data", r
    print("PASS: missing category data on either side reported honestly, never crashes "
          "or guesses a classification")


if __name__ == "__main__":
    test_identical_leaf_is_direct()
    test_same_immediate_parent_is_direct()
    test_same_grandparent_is_indirect()
    test_only_generic_top_level_is_related()
    test_no_overlap_is_irrelevant()
    test_fallback_when_hierarchy_missing()
    test_one_sided_hierarchy_falls_back_too()
    test_batch_classify_and_summary()
    test_insufficient_data_never_crashes()
    print("\nALL COMPETITIVE CLASSIFIER OFFLINE TESTS PASSED")
