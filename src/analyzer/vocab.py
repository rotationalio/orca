from src.corpus import OnlineTextCorpus

from collections import Counter
from gensim.corpora import Dictionary

class VocabAnalyzer():
    """
    VocabAnalyzer computes vocab statistics given a gensim Dictionary.
    """

    def __init__(self, dictionary):
        if not isinstance(dictionary, Dictionary):
            raise ValueError("corpus must be a gensim Dictionary.")
        self.dictionary = dictionary

    def corpus_statistics(self):
        """
        Compute overall vocab statistics for the corpus and return the results.
        """
        stats = {}
        stats["num_docs"] = self.dictionary.num_docs
        stats["num_words"] = self.dictionary.num_pos
        stats["unique_words"] = len(self.dictionary.cfs)
        if self.dictionary.num_docs > 0:
            stats["words_per_doc"] = self.dictionary.num_pos / self.dictionary.num_docs
        else:
            stats["words_per_doc"] = 0

        rare_tokens = []
        word_freq = Counter()
        for token, id in self.dictionary.token2id.items():
            if self.dictionary.cfs[id] == 1:
                rare_tokens.append(token)
            word_freq[token] = self.dictionary.cfs[id]

        stats["most_common"] = word_freq.most_common(10)
        stats["rare_words"] = rare_tokens
        return stats

    def word_statistics(self, word):
        """
        Compute vocab statistics for the given word.
        """
        if word not in self.dictionary.token2id:
            return None
        
        stats = {}
        stats["corpus_count"] = self.dictionary.cfs[self.dictionary.token2id[word]]
        stats["doc_count"] = self.dictionary.dfs[self.dictionary.token2id[word]]
        return stats