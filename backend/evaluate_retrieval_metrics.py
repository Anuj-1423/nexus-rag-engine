import argparse
import asyncio
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

os.chdir(Path(__file__).resolve().parent)

import rag
from langchain_core.embeddings import Embeddings


FALLBACK_CORPUS = {
    "alpha.txt": "The enterprise password policy requires at least 12 characters and MFA for all users.",
    "security_policy.txt": "Enterprise security policy mandates MFA, strong passwords, and device compliance checks for all accounts.",
    "device_compliance.txt": "Device compliance policy requires approved endpoints, password rotation, and access verification before login.",
    "beta.txt": "The payment report was uploaded in Q4 and finance approved the project budget summary.",
    "finance_q4_summary.txt": "Q4 finance summary includes uploaded invoice totals, approved budget, and final spending review.",
    "budget_report.txt": "The budget report for Q4 includes revenue tracking, projected spend, and approved expense notes.",
    "gamma.txt": "The support team confirms the enterprise plan includes 24/7 assistance and uptime guarantees.",
    "service_level_agreement.txt": "The enterprise service level agreement covers 24/7 support, uptime guarantees, and priority incident response.",
    "support_coverage.txt": "Support coverage includes help desk access, proactive maintenance, and guaranteed response windows.",
    "delta.txt": "The onboarding checklist includes device setup, account verification, and a welcome email.",
    "welcome_checklist.txt": "Welcome checklist covers onboarding tasks like account setup, device configuration, and verification guidance.",
    "account_setup_guide.txt": "Onboarding account setup guide explains verification, profile completion, and first-login device configuration.",
}


class DeterministicEmbedding(Embeddings):
    """Small deterministic embedding used only for local metric evaluation."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _vectorize(self, text: str):
        cleaned = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * self.dimensions
        if not cleaned:
            return vector

        for token in cleaned:
            h = int.from_bytes(token.encode("utf-8"), byteorder="big", signed=False) % self.dimensions
            vector[h] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts):
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text):
        return self._vectorize(text)


def load_benchmark(path: str | None) -> List[Dict[str, Any]]:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        if isinstance(data, list):
            return data
    return []


def ensure_fallback_corpus() -> None:
    """Create a clean temporary evaluation index so the benchmark corpus is always the source of truth."""
    temp_root = Path(tempfile.mkdtemp(prefix="rag_eval_"))
    rag.BASE_CHROMA_PATH = str(temp_root / "storage" / "chroma")
    rag.CACHE_PATH = str(temp_root / "query_cache.json")

    global_index = Path(rag.get_index_path("global"))
    if global_index.exists():
        shutil.rmtree(global_index, ignore_errors=True)

    rag._query_cache = {}
    rag._bm25_cache = {}
    rag._hf_embeddings = DeterministicEmbedding()

    for filename, text in FALLBACK_CORPUS.items():
        rag.ingest_document(text.encode("utf-8"), filename, scope="global")


def extract_document_names(results: Iterable[Any]) -> List[str]:
    names: List[str] = []
    for item in results:
        doc = item[0] if isinstance(item, tuple) else item
        name = getattr(doc, "metadata", {}).get("filename")
        if name:
            names.append(name)
    return names


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / k


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    for idx, name in enumerate(retrieved, start=1):
        if name in relevant:
            return 1.0 / idx
    return 0.0


async def evaluate_queries(benchmark: List[Dict[str, Any]], k_values: Tuple[int, ...]) -> None:
    if not benchmark:
        print("No benchmark dataset found. Add a JSON file with queries and relevant_files.")
        return

    metrics = {k: [] for k in k_values}
    mrr_values: List[float] = []

    for case in benchmark:
        query = case.get("query")
        relevant = set(case.get("relevant_files", []))
        mode = case.get("mode", "combined")
        user_email = case.get("user_email")

        if not query:
            continue

        results = await rag.retrieve_context(query, mode=mode, user_email=user_email)
        retrieved = extract_document_names(results)

        print(f"\nQUERY: {query}")
        print(f"Relevant: {sorted(relevant)}")
        print(f"Retrieved: {retrieved[:10]}")

        for k in k_values:
            p = precision_at_k(relevant, retrieved, k)
            r = recall_at_k(relevant, retrieved, k)
            metrics[k].append((p, r))
            print(f"  Precision@{k}: {p:.3f}")
            print(f"  Recall@{k}: {r:.3f}")

        mrr_values.append(reciprocal_rank(relevant, retrieved))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k in k_values:
        avg_p = sum(v[0] for v in metrics[k]) / len(metrics[k]) if metrics[k] else 0.0
        avg_r = sum(v[1] for v in metrics[k]) / len(metrics[k]) if metrics[k] else 0.0
        print(f"Average Precision@{k}: {avg_p:.3f}")
        print(f"Average Recall@{k}: {avg_r:.3f}")

    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0
    print(f"MRR: {avg_mrr:.3f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval metrics.")
    parser.add_argument("--dataset", default="eval_dataset.json", help="Path to JSON benchmark file.")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10], help="Cutoffs for Precision@k and Recall@k.")
    args = parser.parse_args()

    print("RAG retrieval evaluation")
    print("=" * 70)

    ensure_fallback_corpus()

    dataset_path = Path(args.dataset)
    benchmark = load_benchmark(str(dataset_path))
    if not benchmark:
        benchmark = load_benchmark(str(Path(__file__).with_name("eval_dataset.json")))

    if not benchmark:
        benchmark = [
            {"query": "What is the enterprise password policy?", "mode": "enterprise", "user_email": "demo@example.com", "relevant_files": ["alpha.txt"]},
            {"query": "What was uploaded in Q4?", "mode": "enterprise", "user_email": "demo@example.com", "relevant_files": ["beta.txt"]},
            {"query": "What does the enterprise plan include?", "mode": "enterprise", "user_email": "demo@example.com", "relevant_files": ["gamma.txt"]},
            {"query": "What is included in onboarding?", "mode": "enterprise", "user_email": "demo@example.com", "relevant_files": ["delta.txt"]},
        ]

    await evaluate_queries(benchmark, tuple(sorted(set(args.k))))


if __name__ == "__main__":
    asyncio.run(main())
