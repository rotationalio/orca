from src.analyzer.engines.engine import ModelingEngine

import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

class SklearnEngine(ModelingEngine):
    """
    GensimEngine implements the methods for fitting and evaluating gensim models on a
    corpus.
    """
    def __init__(self, min_topics, max_topics):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.grid = None
        self.model = None
        self.vectorizer = None

    def fit(self, type, vectorizer, corpus, **kwargs):
        """
        Fits a set of models on the provided corpus.
        """
        if type == "lda":
            self._fit_lda(corpus, **kwargs)
        elif type == "nmf":
            self._fit_nmf(corpus, **kwargs)
        else:
            raise ValueError("model must be either lda or nmf.")
        self.corpus = corpus
        self.vectorizer = vectorizer

    def _fit_lda(self, corpus, **kwargs):
        """
        Fits a set of models on the provided corpus.
        """
        print("fitting lda models on topic range {} to {}".format(self.min_topics, self.max_topics))
        grid_params = {'n_components': list(range(self.min_topics, self.max_topics + 1))}
        model = LatentDirichletAllocation(learning_method="online", **kwargs)
        self.grid = GridSearchCV(model, param_grid=grid_params)
        self.grid.fit(corpus)
        self.model = self.grid.best_estimator_
        print("best model params: {}".format(self.grid.best_params_))
        print("best model score: {}".format(self.grid.best_score_))
        print("perplexity: {}".format(self.model.perplexity(corpus)))

    def _fit_nmf(self, corpus, **kwargs):
        """
        Fits a set of models on the provided corpus.
        """
        print("fitting nmf models on topic range {} to {}".format(self.min_topics, self.max_topics))
        min_error = float("inf")
        for num_topics in range(self.min_topics, self.max_topics + 1):
            model = NMF(n_components=num_topics, **kwargs)
            model.fit(corpus)
            print("num topics: {} error {}".format(num_topics, model.reconstruction_err_))
            if model.reconstruction_err_ < min_error:
                min_error = model.reconstruction_err_
                self.model = model
        print("best model params: {}".format(self.model.get_params()))
        print("best model score: {}".format(self.model.reconstruction_err_))

    def update(self, documents):
        """
        Updates the set of models with a stream of new corpus documents.
        """
        self.grid.fit(documents)
        self.model = self.grid.best_estimator_

    def topics(self):
        """
        Return the discovered topics and evaluation metrics for the set of current models.
        """
        topics = {}
        topic_list = []
        feature_names = self.vectorizer.get_feature_names()
        for idx, topic in enumerate(self.model.components_):
            top_features_idx = topic.argsort()[:-10:-1]
            top_features = [(feature_names[i], topic[i]) for i in top_features_idx]
            topics[idx] = top_features
        return topics