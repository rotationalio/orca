import os
import shutil
from gensim.corpora import Dictionary

CORPUS_DIR = "corpus"
DICT_NAME = "dict"

class OnlineTextCorpus():
    """
    OnlineTextCorpus is a wrapper around gensim corpus data structures which is
    expected to be updated on an event-driven basis (e.g., when new data is received).
    The corpus is frequently checkpointed to disk to allow for data recovery.
    TODO: Add a dense matrix format to support NLP transformations.
    """
    def __init__(self, dir, num_checkpoints=5):
        if num_checkpoints < 1:
            raise ValueError("num_checkpoints must be at least 1.")
        self.dir = dir
        self.num_checkpoints = num_checkpoints
        self.dictionary = None
        self.version = 0

    def _corpus_path(self):
        return os.path.join(self.dir, CORPUS_DIR + "_" + str(self.version))

    def _dict_path(self):
        return os.path.join(self._corpus_path(), DICT_NAME)

    def get_dictionary(self):
        """
        Return the current dictionary, loading the most recent version from disk if
        necessary.
        """
        if self.dictionary is None:
            self.load()
        return self.dictionary

    def save(self):
        """
        Save the current corpus to disk.
        """
        if self.version > 0:
            os.makedirs(self._corpus_path(), exist_ok=True)
            self.dictionary.save(self._dict_path())
            for f in os.listdir(self.dir):
                # Remove older corpus versions.
                if os.path.isdir(os.path.join(self.dir, f)) and f.startswith(CORPUS_DIR + "_"):
                    version = int(f.split("_")[1])
                    if version <= self.version - self.num_checkpoints:
                        shutil.rmtree(os.path.join(self.dir, f))
        else:
            raise ValueError("Cannot save an empty corpus.")

    def load(self):
        """
        Load the latest version of the corpus from disk.
        """
        if self.version > 0:
            self.dictionary = Dictionary.load(self._dict_path())
        else:
            # Search for the latest version of the corpus.
            latest_version = 0
            for f in os.listdir(self.dir):
                if f.startswith(CORPUS_DIR + "_"):
                    version = int(f.split("_")[1])
                    if version > latest_version:
                        latest_version = version
            if latest_version > 0:
                self.version = latest_version
                self.dictionary = Dictionary.load(self._dict_path())
            else:
                self.dictionary = Dictionary()
                self.version = 0

    def add_documents(self, documents):
        """
        Add a list of documents to the corpus, incrementing the version number and
        checkpointing the results.
        """
        if self.dictionary is None:
            self.load()
        self.dictionary.add_documents(documents)
        self.version += 1
        self.save()