"""
Unit test for the metadata filter service's degrade-to-none behavior when
no section candidates or LLM matches are found — this is the safety-net
path that must never crash the pipeline.
"""

from unittest.mock import MagicMock

from rag_harness.application.retrieval.metadata_filter_service import MetadataFilterService


def test_build_filter_returns_none_when_no_candidates():
    selector = MagicMock()
    selector.get_candidates.return_value = []
    filter_llm = MagicMock()

    service = MetadataFilterService(selector, filter_llm)
    result_filter, matched = service.build_filter("some query")

    assert result_filter is None
    assert matched == []
    filter_llm.select_sections.assert_not_called()


def test_build_filter_returns_none_when_llm_matches_nothing():
    selector = MagicMock()
    selector.get_candidates.return_value = ["Section A", "Section B"]
    filter_llm = MagicMock()
    filter_llm.select_sections.return_value = []

    service = MetadataFilterService(selector, filter_llm)
    result_filter, matched = service.build_filter("some query")

    assert result_filter is None
    assert matched == []


def test_build_filter_returns_single_match_filter():
    selector = MagicMock()
    selector.get_candidates.return_value = ["Section A", "Section B"]
    filter_llm = MagicMock()
    filter_llm.select_sections.return_value = ["Section A"]

    service = MetadataFilterService(selector, filter_llm)
    result_filter, matched = service.build_filter("some query")

    assert result_filter == {"section_title": "Section A"}
    assert matched == ["Section A"]