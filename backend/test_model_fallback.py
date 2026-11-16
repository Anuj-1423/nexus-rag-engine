import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document

import rag


class GeminiModelFallbackTests(unittest.IsolatedAsyncioTestCase):
    async def test_groq_failure_falls_back_to_gemini(self):
        client = SimpleNamespace(models=SimpleNamespace())
        attempted_models = []

        def generate_content(*, model, **kwargs):
            attempted_models.append(model)
            return SimpleNamespace(text="The address is listed in the document.")

        client.models.generate_content = generate_content
        document = Document(
            page_content="Anuj's address is Example Road.",
            metadata={"filename": "contact.txt", "scope": "global"},
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = os.path.join(temp_dir, "query_cache.json")
            with (
                patch.dict(os.environ, {
                    "GOOGLE_API_KEY": "test-key",
                    "GROQ_API_KEY": "test-groq-key",
                }),
                patch.object(rag.genai, "Client", return_value=client),
                patch.object(
                    rag,
                    "generate_with_groq",
                    side_effect=RuntimeError("Groq unavailable"),
                ) as groq_generation,
                patch.object(
                    rag,
                    "retrieve_context",
                    return_value=[(document, 0.1)],
                ),
                patch.object(rag, "CACHE_PATH", cache_path),
                patch.object(
                    rag,
                    "GEMINI_FALLBACK_MODEL",
                    "gemini-3.5-flash-lite",
                ),
                patch.dict(rag._query_cache, {}, clear=True),
            ):
                result = await rag.generate_rag_response("What is the address?")
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                saved_cache = json.load(cache_file)

        self.assertEqual(
            attempted_models,
            ["gemini-3.5-flash-lite"],
        )
        self.assertEqual(result["answer"], "The address is listed in the document.")
        self.assertEqual(result["provider"], "gemini")
        groq_generation.assert_called_once()
        self.assertIn("What is the address?", next(iter(saved_cache)))

    async def test_groq_success_is_primary_and_skips_gemini(self):
        document = Document(
            page_content="Anuj's address is Example Road.",
            metadata={"filename": "contact.txt", "scope": "global"},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.dict(os.environ, {
                    "GOOGLE_API_KEY": "test-key",
                    "GROQ_API_KEY": "test-groq-key",
                }),
                patch.object(
                    rag,
                    "retrieve_context",
                    return_value=[(document, 0.1)],
                ),
                patch.object(
                    rag,
                    "generate_with_groq",
                    return_value="The address is Example Road.",
                ) as groq_generation,
                patch.object(rag.genai, "Client") as gemini_client,
                patch.object(rag, "CACHE_PATH", os.path.join(temp_dir, "query_cache.json")),
                patch.dict(rag._query_cache, {}, clear=True),
            ):
                result = await rag.generate_rag_response("What is the address?")

        self.assertEqual(result["answer"], "The address is Example Road.")
        self.assertEqual(result["provider"], "groq")
        self.assertEqual(result["model"], rag.GROQ_MODEL)
        groq_generation.assert_called_once()
        gemini_client.assert_not_called()

    def test_vercel_runtime_data_uses_writable_tmp_directory(self):
        with patch.dict(os.environ, {"VERCEL": "1"}):
            runtime_dir = rag.get_runtime_storage_dir()

        self.assertEqual(runtime_dir.replace("\\", "/"), "/tmp/nexus-rag")


if __name__ == "__main__":
    unittest.main()
