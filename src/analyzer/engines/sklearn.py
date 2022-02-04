from sklearn.base import BaseEstimator, TransformerMixin
from src.analyzer.engines.engine import ModelingEngine

from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

class SklearnEngine(ModelingEngine, BaseEstimator, TransformerMixin):
    """
    SklearnEngine implements the methods for fitting and evaluating sklearn models on a
    corpus. This class implements the fit(), transform(), and predict() methods in
    order to support use in sklearn pipelines.
    """
    def __init__(self, engine="lda", vectorizer=None, min_topics=5, max_topics=10, **kwargs):
        if self.engine not in ["lda", "nmf"]:
            raise ValueError("engine must be either 'lda' or 'nmf'")
        self.engine = engine
        # TODO: Can we make the vectorizer part of the pipeline so we don't have to
        #       pass it in here?
        self.vectorizer = vectorizer
        self.engine_opts = kwargs
        self.min_topics = min_topics
        self.max_topics = max_topics

    def fit(self, corpus):
        """
        Fits a set of models on the provided corpus, this conforms to a fit() method in
        the sklearn API.
        """
        if self.engine == "lda":
            self._fit_lda(corpus, **self.engine_opts)
        elif self.engine == "nmf":
            self._fit_nmf(corpus, **self.engine_opts)
        else:
            raise ValueError("engine must be either 'lda' or 'nmf'")
        self.corpus_ = corpus
        return self

    def _fit_lda(self, corpus, **kwargs):
        """
        Fits a set of models on the provided corpus.
        """
        print("fitting lda models on topic range {} to {}".format(self.min_topics, self.max_topics))
        grid_params = {'n_components': list(range(self.min_topics, self.max_topics + 1))}
        model = LatentDirichletAllocation(learning_method="online", **kwargs)
        self.grid_ = GridSearchCV(model, param_grid=grid_params)
        self.grid_.fit(corpus)
        self.model_ = self.grid_.best_estimator_
        print("best model params: {}".format(self.grid_.best_params_))
        print("best model score: {}".format(self.grid_.best_score_))
        print("perplexity: {}".format(self.model_.perplexity(corpus)))

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
                self.model_ = model
        print("best model params: {}".format(self.model_.get_params()))
        print("best model score: {}".format(self.model_.reconstruction_err_))

    def transform(self, documents):
        """
        Implementation of the standard transform() method of the sklearn API.
        """
        self.update(self, documents)

    def update(self, documents):
        """
        Updates the set of models with a stream of new corpus documents.
        """
        self.grid_.fit(documents)
        self.model_ = self.grid_.best_estimator_

    def predict(self):
        """
        Implementation of the standard predict() method of the sklearn API.
        """
        return self.topics()

    def topics(self):
        """
        Return the discovered topics and evaluation metrics for the set of current models.
        """
        topics = {}
        feature_names = self.vectorizer.get_feature_names()
        for idx, topic in enumerate(self.model_.components_):
            top_features_idx = topic.argsort()[:-10:-1]
            top_features = [(feature_names[i], topic[i]) for i in top_features_idx]
            topics[idx] = top_features
        return topics