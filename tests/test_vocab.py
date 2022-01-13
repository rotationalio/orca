from src.analyzer.vocab import VocabAnalyzer

import pytest
from gensim.corpora import Dictionary

class TestVocabAnalyzer():
    """
    Tests for the VocabAnalyzer class.
    """

    @pytest.mark.parametrize(
        "documents, expected",
        [
            ([], {
                "num_docs": 0,
                "num_words": 0,
                "unique_words": 0,
                "words_per_doc": 0,
                "most_common": [],
                "rare_words": 0,
            }),
            ([
                ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "in", "forest"],
                ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "in", "forest"],
                ["quick", "brown", "fox", "jumps", "over", "lazy", "dog", "in", "forest"],
                ["brown", "fox", "jumps", "over", "lazy", "dog", "in", "forest"],
                ["fox", "jumps", "over", "lazy", "dog", "in", "forest"],
                ["jumps", "over", "lazy", "dog", "in", "forest"],
                ["over", "lazy", "dog", "in", "forest"],
                ["lazy", "dog", "in", "forest"],
                ["dog", "in", "forest"],
                ["in", "forest"],
                ["forest"],
                ["rare", "words"],
              ], {
                "num_docs": 12,
                "num_words": 67,
                "unique_words": 12,
                "words_per_doc": 67 / 12,
                "most_common": [("forest", 11), ("in", 10), ("dog", 9), ("lazy", 8), ("over", 7), ("jumps", 6), ("fox", 5), ("brown", 4), ("quick", 3), ("the", 2)],
                "rare_words": 2,
            }),
        ]
    )
    def test_corpus_statistics(self, documents, expected):
        """
        Test that corpus_statistics() returns the correct results.
        """
        vocab = VocabAnalyzer(Dictionary(documents))
        assert vocab.corpus_statistics() == expected

    @pytest.mark.parametrize(
        "documents, word, expected",
        [
            ([], "the", None),
            ([
                ["the", "quick", "brown", "fox", "jumps", "over", "the"],
                ["lazy", "dog"],
             ], "the", {
                "corpus_count": 2,
                "doc_count": 1,
            }),
        ]
    )
    def test_word_statistics(self, documents, word, expected):
        """
        Test that word_statistics() returns the correct results.
        """
        vocab = VocabAnalyzer(Dictionary(documents))
        assert vocab.word_statistics(word) == expected