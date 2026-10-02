"""Tests for the optional `embed_context_fields` document setting.

These import the real `webiks_hebrew_ragbot` package (the pre-existing tests import a
`ragbot` package from a `src/` directory that does not exist in this repository).
"""
import importlib
import json
import os
import sys
from unittest.mock import MagicMock

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

BASE_CONFIG = {
    "identifier_field": "doc_id",
    "saved_fields": {"title": "text", "doc_id": "integer", "link": "text", "content": "text"},
    "field_for_llm": "content",
    "model_name": "Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0",
    "field_to_embed": "content",
}


def _load_engine(tmp_path, extra):
    """Reload document + engine modules against a config file with `extra` keys."""
    cfg = tmp_path / "doc-config.json"
    cfg.write_text(json.dumps({**BASE_CONFIG, **extra}, ensure_ascii=False), encoding="utf-8")
    os.environ["DOCUMENT_DEFINITION_CONFIG"] = str(cfg)
    import webiks_hebrew_ragbot.document as document
    importlib.reload(document)
    import webiks_hebrew_ragbot.engine as engine
    importlib.reload(engine)
    return engine


def _engine_with_fake_model(engine_module):
    fake_model = MagicMock()
    fake_model.encode.side_effect = lambda text: [len(text)]      # vector depends on the text
    return engine_module.Engine(llms_client=MagicMock(), elastic_model=MagicMock(), retrieval_model=fake_model), fake_model


def test_default_embeds_content_only(tmp_path):
    engine = _load_engine(tmp_path, {})
    eng, model = _engine_with_fake_model(engine)
    doc = {"doc_id": 1, "title": "קצבת זיקנה", "content": "אזרחים ותיקים זכאים לקצבה."}
    assert eng.text_to_embed(doc) == doc["content"]
    eng.create_paragraphs([doc])
    model.encode.assert_called_once_with(doc["content"])


def test_title_is_prepended_when_configured(tmp_path):
    engine = _load_engine(tmp_path, {"embed_context_fields": ["title"]})
    eng, model = _engine_with_fake_model(engine)
    doc = {"doc_id": 1, "title": "קצבת זיקנה", "content": "אזרחים ותיקים זכאים לקצבה."}
    assert eng.text_to_embed(doc) == "קצבת זיקנה\nאזרחים ותיקים זכאים לקצבה."
    eng.update_docs([doc])
    model.encode.assert_called_once_with("קצבת זיקנה\nאזרחים ותיקים זכאים לקצבה.")
    # stored fields and the vector field name are unchanged
    assert doc["content"] == "אזרחים ותיקים זכאים לקצבה."
    assert "content_Webiks_Hebrew_RAGbot_KolZchut_QA_Embedder_v1.0_vectors" in doc


def test_missing_context_value_is_skipped(tmp_path):
    engine = _load_engine(tmp_path, {"embed_context_fields": ["title"]})
    eng, _ = _engine_with_fake_model(engine)
    assert eng.text_to_embed({"doc_id": 2, "title": "", "content": "גוף"}) == "גוף"
    assert eng.text_to_embed({"doc_id": 3, "content": "גוף"}) == "גוף"


def test_query_path_is_not_affected(tmp_path):
    engine = _load_engine(tmp_path, {"embed_context_fields": ["title"]})
    eng, model = _engine_with_fake_model(engine)
    eng.elastic_model.search = MagicMock(return_value=[{"_source": {"doc_id": 1, "title": "t"}}])
    eng.search_documents("מי זכאי?", 3)
    model.encode.assert_called_once_with("מי זכאי?")


def test_unknown_context_field_is_rejected(tmp_path):
    cfg = tmp_path / "bad.json"
    cfg.write_text(json.dumps({**BASE_CONFIG, "embed_context_fields": ["nope"]}), encoding="utf-8")
    os.environ["DOCUMENT_DEFINITION_CONFIG"] = str(cfg)
    import webiks_hebrew_ragbot.document as document
    importlib.reload(document)
    with pytest.raises(ValueError):
        document.initialize_definitions()
