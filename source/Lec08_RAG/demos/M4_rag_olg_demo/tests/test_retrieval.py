from __future__ import annotations

import unittest
import json
from pathlib import Path

from rag_olg.chunking import split_markdown_into_chunks
from rag_olg.generation import extractive_answer
from rag_olg.retrieval import build_tfidf_index, search
from rag_olg.llm import hits_anchor
from questions import QUESTIONS


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source" / "Spear-Young_OLG_final_preprint.md"


class RagOlgTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        markdown = SOURCE.read_text(encoding="utf-8")
        cls.chunks = split_markdown_into_chunks(
            markdown,
            source_name="data/source/Spear-Young_OLG_final_preprint.md",
        )
        cls.index = build_tfidf_index(cls.chunks)

    def test_chunks_are_created(self) -> None:
        self.assertGreater(len(self.chunks), 100)
        self.assertLess(len(self.chunks), 800)
        self.assertEqual(self.chunks[0].start_line, 1)

    def test_olg_ila_query_retrieves_relevant_passage(self) -> None:
        results = search(
            self.chunks,
            self.index,
            "How do Spear and Young distinguish OLG from ILA models?",
            top_k=5,
        )
        joined = " ".join(result.chunk.text for result in results).lower()
        self.assertGreater(results[0].score, 0.05)
        self.assertIn("olg", joined)
        self.assertTrue("infinite lived" in joined or "ila" in joined)

    def test_unsupported_question_refuses(self) -> None:
        results = search(
            self.chunks,
            self.index,
            "What does this paper say about zebra nebula harmonica?",
            top_k=5,
        )
        answer = extractive_answer(
            "What does this paper say about zebra nebula harmonica?",
            results,
        )
        self.assertIn("do not have enough retrieved evidence", answer)

    def test_curated_questions_have_retrieval_diagnostics(self) -> None:
        for item in QUESTIONS:
            with self.subTest(question_id=item["id"]):
                results = search(self.chunks, self.index, item["q"], top_k=5)
                self.assertTrue(results)
                if item.get("expect_refusal"):
                    answer = extractive_answer(item["q"], results)
                    self.assertIn("do not have enough retrieved evidence", answer)
                else:
                    self.assertGreater(results[0].score, 0.035)
                    self.assertGreaterEqual(
                        len(hits_anchor(results, item.get("anchors", []), top_n=5)),
                        1,
                    )

    def test_notebook_is_clean_and_present(self) -> None:
        notebook_path = ROOT / "demo.ipynb"
        self.assertTrue(notebook_path.exists())
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(notebook["cells"]), 10)
        output_count = sum(len(cell.get("outputs", [])) for cell in notebook["cells"])
        self.assertEqual(output_count, 0)


if __name__ == "__main__":
    unittest.main()
