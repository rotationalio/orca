import os
import shutil
from gensim.corpora import Dictionary
from gensim.corpora import MmCorpus
from gensim.corpora import HashDictionary

CORPUS_DIR = "corpus"
DICT_NAME = "dict"
MATRIX_NAME = "corpus.mm"

class OnlineTextCorpus():
    """
    OnlineTextCorpus is a wrapper around gensim corpus data structures which is
    expected to be updated on an event-driven basis (e.g., when new data is received).
    The corpus is frequently checkpointed to disk to allow for data recovery.
    """
    def __init__(self, dir, hash_dictionary=False, num_checkpoints=5):
        if num_checkpoints < 1:
            raise ValueError("num_checkpoints must be at least 1.")
        self.dir = dir
        self.hash_dictionary = hash_dictionary
        self.num_checkpoints = num_checkpoints
        self.dictionary = None
        self.mm = None
        self.version = 0

    def _corpus_path(self):
        return os.path.join(self.dir, CORPUS_DIR + "_" + str(self.version))

    def _dict_path(self):
        return os.path.join(self._corpus_path(), DICT_NAME)

    def _matrix_path(self):
        return os.path.join(self._corpus_path(), MATRIX_NAME)

    def _dictionary(self):
        if self.hash_dictionary:
            return HashDictionary
        else:
            return Dictionary

    def _load_corpus(self):
        """
        Load the dictionary into memory and set the path to the streamed corpus.
        """
        self.dictionary = self._dictionary().load(self._dict_path())
        self.mm = MmCorpus(self._matrix_path())

    def get_dictionary(self):
        """
        Return the current dictionary, loading the most recent version from disk if
        necessary.
        """
        if self.dictionary is None:
            self.load()
        return self.dictionary

    def load(self):
        """
        Load the latest version of the corpus from disk.
        """
        if self.version > 0:
            self._load_corpus()
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
                self._load_corpus()
            else:
                self.dictionary = self._dictionary()()
                self.version = 0

    def iter_corpus(self, documents=[]):
        """
        Returns a generator which iterates through the current corpus and then the
        given document stream. This method is used internally to update the
        dictionary and corpus when new documents are added, but if the documents
        argument is omitted, then this can be used to simply iterate through the
        documents in the corpus.
        """
        if self.mm is not None:
            for doc in self.mm:
                yield doc
        for doc in documents:
            yield self.dictionary.doc2bow(doc, allow_update=True)

    def add_documents(self, documents):
        """
        Add a stream of documents to the corpus, incrementing the version number and
        checkpointing the results.
        """
        if self.dictionary is None or self.mm is None:
            self.load()
        self.version += 1

        # Save the new version of the corpus to a new directory.
        # FIXME: This requires a full scan of the current corpus which does not scale.
        os.makedirs(self._corpus_path(), exist_ok=True)
        MmCorpus.serialize(self._matrix_path(), self.iter_corpus(documents=documents))
        self.mm = MmCorpus(self._matrix_path())
        self.dictionary.save(self._dict_path())
        for f in os.listdir(self.dir):
            # Remove older corpus versions.
            if os.path.isdir(os.path.join(self.dir, f)) and f.startswith(CORPUS_DIR + "_"):
                version = int(f.split("_")[1])
                if version <= self.version - self.num_checkpoints:
                    shutil.rmtree(os.path.join(self.dir, f))