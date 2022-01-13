from src.corpus import OnlineTextCorpus

import os
import pytest
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus

class TestOnlineTextCorpus():
    """
    Tests for the OnlineTextCorpus class.
    """

    @pytest.mark.parametrize(
        "num_checkpoints",
        [-1, 0]
    )
    def test_invalid_init(self, tmpdir, num_checkpoints):
        """
        Test that the class raises an exception when initialized with invalid
        parameters.
        """
        with pytest.raises(ValueError):
            corpus = OnlineTextCorpus(tmpdir, num_checkpoints=num_checkpoints)

    @pytest.mark.parametrize(
        "num_checkpoints",
        [1, 2, 5]
    )
    def test_load(self, tmpdir, num_checkpoints):
        """
        Test that load() loads the most recent version of the corpus from disk.
        """
        dictionary = Dictionary()
        for i in range(num_checkpoints):
            version = str(i + 1)
            corpus_path = os.path.join(tmpdir, "corpus_" + version)
            os.mkdir(corpus_path)
            dictionary.add_documents([[version]])
            dictionary.save(os.path.join(corpus_path, "dict"))
            mm_path = os.path.join(corpus_path, "corpus.mm")
            MmCorpus.serialize(mm_path, [dictionary.doc2bow([version])])
        
        corpus = OnlineTextCorpus(tmpdir, num_checkpoints=num_checkpoints)

        # When there is a known version, that version should be loaded from disk.
        corpus.version = 1
        corpus.load()
        assert str(corpus.version) in corpus.dictionary.token2id

        # When the version is unknown, the latest version is used.
        corpus.version = 0
        corpus.load()
        assert corpus.version == num_checkpoints
        assert str(num_checkpoints) in corpus.dictionary.token2id

    @pytest.mark.parametrize(
        "documents, expected_words",
        [
            ([[]], 2),
            ([["hello", "world"]], 2),
            ([["the", "quick", "brown", "fox"],
              ["jumped", "over", "the", "lazy", "dog"]], 10),
        ]
    )
    def test_add_documents(self, tmpdir, documents, expected_words):
        """
        Test that add_documents() adds documents to the corpus.
        """
        corpus = OnlineTextCorpus(tmpdir)
        corpus.add_documents([["hello", "world"]])
        assert corpus.version == 1
        assert corpus.dictionary.num_docs == 1
        assert len(list(corpus.mm)) == 1

        corpus.add_documents(documents)
        assert corpus.version == 2
        assert corpus.dictionary.num_docs == len(documents) + 1
        assert len(corpus.dictionary.token2id) == expected_words
        assert len(list(corpus.mm)) == len(documents) + 1

    @pytest.mark.parametrize(
        "initial, additional, expected_words",
        [
            ([], [], []),
            ([], [["hello", "world"]], [2]),
            ([["the", "quick", "brown", "fox"],
              ["jumped", "over", "the", "lazy", "dog"]], [], [4, 5]),
            ([["the", "quick", "brown"]], [["fox"], ["jumped"]], [3, 1, 1]),
        ]
    )
    def test_iter_corpus(self, tmpdir, initial, additional, expected_words):
        """
        Test that iter_corpus() returns a generator that yields the documents in
        the corpus.
        """
        corpus = OnlineTextCorpus(tmpdir)
        corpus.add_documents(initial)

        # Iterate over the corpus and ensure that the documents are yielded in
        # the correct order.
        counts = []
        for document in corpus.iter_corpus(documents=additional):
            counts.append(len(document))
        assert counts == expected_words