from src.corpus import OnlineTextCorpus

import os
import pytest
from gensim.corpora import Dictionary

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

    def test_save_invalid(self, tmpdir):
        """
        Test that save() raises an exception when the corpus is empty.
        """
        corpus = OnlineTextCorpus(tmpdir)
        corpus.dictionary = Dictionary()
        with pytest.raises(ValueError):
            corpus.save()

    @pytest.mark.parametrize(
        "num_checkpoints",
        [1, 2, 5]
    )
    def test_save(self, tmpdir, num_checkpoints):
        """
        Test that save() saves the corpus to disk.
        """
        corpus = OnlineTextCorpus(tmpdir, num_checkpoints=num_checkpoints)
        corpus.version = 1
        corpus.dictionary = Dictionary([["hello", "world"]])
        corpus.save()
        corpus_path = os.path.join(tmpdir, "corpus_1")
        assert os.path.isdir(corpus_path)
        dict_path = os.path.join(corpus_path, "dict")
        assert os.path.isfile(dict_path)

        # Save a few more times to trigger version cleanup.
        for i in range(num_checkpoints):
            corpus.version += 1
            corpus.save()
        
        # Check that the older versions of the corpus have been removed.
        dirs = os.listdir(tmpdir)
        assert len(dirs) == num_checkpoints
        for d in dirs:
            assert d.startswith("corpus_")
            assert int(d.split("_")[1]) <= corpus.version
            assert os.path.isfile(os.path.join(tmpdir, d, "dict"))

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
        corpus.add_documents(documents)
        assert corpus.version == 2
        assert corpus.dictionary.num_docs == len(documents) + 1
        assert len(corpus.dictionary.token2id) == expected_words