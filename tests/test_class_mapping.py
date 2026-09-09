from src.preprocessing.class_mapping import (
    ALLOWED_TARGET_CLASSES,
    EXPECTED_SOURCE_LABELS,
    SOURCE_TO_TARGET,
    UNRESOLVED_SOURCE_LABELS,
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


def test_exactly_eight_allowed_target_classes_exist():
    assert len(ALLOWED_TARGET_CLASSES) == 8
    assert len(set(ALLOWED_TARGET_CLASSES)) == 8


def test_no_resolved_source_label_maps_to_none():
    assert all(map_label(label) is not None for label in SOURCE_TO_TARGET)