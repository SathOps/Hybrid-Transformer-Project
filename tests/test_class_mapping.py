from src.preprocessing.class_mapping import (
    ACTIVE_TARGET_CLASSES,
    ALLOWED_TARGET_CLASSES,
    CANONICAL_TARGET_CLASSES,
    EXPECTED_SOURCE_LABELS,
    NUM_ACTIVE_CLASSES,
    NUM_CANONICAL_CLASSES,
    SOURCE_TO_TARGET,
    UNRESOLVED_SOURCE_LABELS,
    active_to_canonical_labels,
    canonical_to_active_labels,
    map_label,
)


def test_every_expected_source_label_is_accounted_for():
    accounted_for = set(SOURCE_TO_TARGET) | set(UNRESOLVED_SOURCE_LABELS)

    assert accounted_for == set(EXPECTED_SOURCE_LABELS)


def test_no_unexpected_source_labels_exist():
    expected = set(EXPECTED_SOURCE_LABELS)

    assert set(SOURCE_TO_TARGET) <= expected
    assert set(UNRESOLVED_SOURCE_LABELS) <= expected


def test_every_resolved_mapping_uses_an_allowed_target():
    assert set(SOURCE_TO_TARGET.values()) <= set(ALLOWED_TARGET_CLASSES)


def test_unresolved_labels_return_none():
    assert map_label("Backdoor_Malware") is None
    assert map_label("VulnerabilityScan") is None


def test_unknown_label_raises_value_error():
    try:
        map_label("UnexpectedLabel")
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected labels must raise ValueError")


def test_canonical_and_active_class_definitions():
    assert len(CANONICAL_TARGET_CLASSES) == 8
    assert NUM_CANONICAL_CLASSES == 8
    assert len(ACTIVE_TARGET_CLASSES) == 7
    assert NUM_ACTIVE_CLASSES == 7
    assert "Recon" in CANONICAL_TARGET_CLASSES
    assert "Recon" not in ACTIVE_TARGET_CLASSES


def test_canonical_to_active_label_mapping():
    # Canonical IDs: 0, 1, 2, 3, 4, 6, 7 -> Active: 0, 1, 2, 3, 4, 5, 6
    canonical = [0, 1, 2, 3, 4, 6, 7]
    active = canonical_to_active_labels(canonical)
    assert list(active) == [0, 1, 2, 3, 4, 5, 6]

    recovered = active_to_canonical_labels(active)
    assert list(recovered) == canonical


def test_no_resolved_source_label_maps_to_none():
    assert all(map_label(label) is not None for label in SOURCE_TO_TARGET)