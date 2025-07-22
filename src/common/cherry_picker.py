import math

from common.semantic_similarity import SemanticSimilarityScorer


class CherryPicker:
    """Finds the most relevant text snippets"""

    def __init__(
        self,
        similarity_scorer: SemanticSimilarityScorer,
        chunk_size: int = 100,
        n_snippets: int = 10,
        snippets_length: int = 400,
        min_similarity: float = 0.1,
    ):
        self.similarity_scorer = similarity_scorer
        self.chunk_size = chunk_size
        self.n_snippets = n_snippets
        self.snippets_length = snippets_length
        self.min_similarity = min_similarity

    def chunk_raw_text(self, text: str) -> list[str]:
        return [
            text[idx : idx + self.chunk_size]
            for idx in range(0, len(text), self.chunk_size)
        ]

    def cherry_pick(
        self,
        question: str,
        text: str,
    ):
        """
        Selects the most relevant text snippets from a given text based on similarity w.r.t the question.
        - Splits the input text into chunks.
        - Computes semantic similarity between the question and each chunk using the similarity scorer.
        - Slides a window over the similarity scores to identify high-scoring spans.
        - Picks the top N most relevant snippets that exceed a similarity threshold.

        Args:
            question (str): The question used to assess relevance.
            text (str): The raw text from which to extract relevant snippets.

        Returns:
            str: A string containing the top-ranked non-overlapping snippets, separated by double newlines.
        """
        chunks_per_snippet = math.ceil(self.snippets_length / self.chunk_size)
        chunks = self.chunk_raw_text(text=text)

        # If not enough chunks for even one snippet, return the full text once
        if len(chunks) < chunks_per_snippet:
            snippet = text[: self.snippets_length]
            return [snippet]

        # Eval semantic similarity w.r.t the question
        similarities = self.similarity_scorer.compute_similarities(
            query=question, docs=chunks
        )

        snippets = []
        for _ in range(0, self.n_snippets):
            best_start_index = 0
            best_score = -math.inf

            # Find the best snippet window based on average similarity
            for j in range(0, len(similarities) - chunks_per_snippet + 1):
                window_scores = similarities[j : j + chunks_per_snippet]
                window_score = sum(window_scores) / len(window_scores)
                if window_score > best_score:
                    best_score = window_score
                    best_start_index = j

            snippet_start_idx = int(best_start_index * self.chunk_size)
            snippet_end_idx = min(snippet_start_idx + self.snippets_length, len(text))

            if best_score > self.min_similarity:
                snippets.append(text[snippet_start_idx:snippet_end_idx])

            # Mask used chunks so they're not reused in future iterations
            for k in range(best_start_index, best_start_index + chunks_per_snippet):
                similarities[k] = -math.inf

        return "\n\n".join(snippets)
