from covsafe.config import canonical_config_hash


def test_hash_is_order_invariant() -> None:
    left = {"b": [2, 1], "a": {"x": 3}}
    right = {"a": {"x": 3}, "b": [2, 1]}
    assert canonical_config_hash(left) == canonical_config_hash(right)


def test_hash_changes_with_value() -> None:
    assert canonical_config_hash({"x": 1}) != canonical_config_hash({"x": 2})
