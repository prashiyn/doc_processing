"""Unit tests for reference extraction post-processing (docs/REFERENCES.md)."""

from __future__ import annotations

from doc_processing.ffp.multimodal.reference_extractor import post_process_references


def test_post_process_dedupes_by_reference_text() -> None:
    raw = [
        {
            "reference_text": "Table 3",
            "reference_type": "TABLE",
            "target_label": "3",
            "confidence": 0.95,
        },
        {
            "reference_text": "table 3",
            "reference_type": "TABLE",
            "target_label": "3",
            "confidence": 0.9,
        },
    ]
    out = post_process_references(raw)
    assert len(out) == 1
    assert out[0]["reference_text"] == "Table 3"


def test_post_process_confidence_floor() -> None:
    raw = [
        {
            "reference_text": "Note 1",
            "reference_type": "NOTE",
            "target_label": "1",
            "confidence": 0.7,
        },
        {
            "reference_text": "Note 2",
            "reference_type": "NOTE",
            "target_label": "2",
            "confidence": 0.71,
        },
    ]
    out = post_process_references(raw)
    assert len(out) == 1
    assert out[0]["reference_text"] == "Note 2"


def test_post_process_unknown_type_maps_to_other() -> None:
    raw = [
        {
            "reference_text": "Something",
            "reference_type": "WEIRD",
            "target_label": "x",
            "confidence": 0.99,
        }
    ]
    out = post_process_references(raw)
    assert len(out) == 1
    assert out[0]["reference_type"] == "OTHER"


def test_post_process_strips_target_label() -> None:
    raw = [
        {
            "reference_text": "Ind AS 115",
            "reference_type": "REGULATION",
            "target_label": "  Ind AS 115  ",
            "confidence": 0.98,
        }
    ]
    out = post_process_references(raw)
    assert out[0]["target_label"] == "Ind AS 115"
