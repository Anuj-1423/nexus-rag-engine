import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

import rag


class GeminiModelFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_retired_primary_model_falls_back_to_supported_model(self):
        client = SimpleNamespace(models=SimpleNamespace())
        attempted_models = []

        def generate_content(*, model, **kwargs):
            attempted_models.append(model)
            if model == "gemini-3.8-flash":
                raise RuntimeError("404 NOT_FOUND: model unavailable")
            return SimpleNamespace(text="The address is listed in the document.")

        client.models.generate_content = generate_content
        document = Document(
            page_content="Anuj's address is Example Road.",
            metadata={"filename": "contact.txt", "scope": "global"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "query_cache.json")
            with (
                patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}),
                patch.object(rag.genai, "Client", return_value=client),
                patch.object(
                    rag,
                    "retrieve_context",
                    return_value=[(document, 0.1)],
                ),
                patch.object(rag, "CACHE_PATH", cache_path),
                patch.object(
                    rag,
                    "GENERATION_MODELS",
                    ("gemini-3.8-flash", "gemini-3.5-flash-lite"),
                ),
                patch.dict(rag._query_cache, {}, clear=True),
            ):
                result = await rag.generate_rag_response("What is the address?")
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                saved_cache = json.load(cache_file)

        self.assertEqual(
            attempted_models,
            ["gemini-3.8-flash", "gemini-3.5-flash-lite"],
        )
        self.assertEqual(result["answer"], "The address is listed in the document.")
        self.assertNotIn("gemini-2.0-flash", attempted_models)
        self.assertIn("What is the address?", next(iter(saved_cache)))

    def test_vercel_runtime_data_uses_writable_tmp_directory(self):
        with patch.dict(os.environ, {"VERCEL": "1"}):
            runtime_dir = rag.get_runtime_storage_dir()

        self.assertEqual(runtime_dir.replace("\\", "/"), "/tmp/nexus-rag")


if __name__ == "__main__":
    unittest.main()
