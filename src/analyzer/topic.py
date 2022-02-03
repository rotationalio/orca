from src.analyzer.engines.gensim import GensimEngine
from src.analyzer.engines.sklearn import SklearnEngine

class TopicAnalyzer():
    """
    TopicAnalyzer supports fitting various topic models on a gensim corpus.
    """
    def __init__(self, engine="gensim", type="lda", dictionary=None, min_topics=5, max_topics=10, **kwargs):
        if engine == "gensim":
            if dictionary is None:
                raise ValueError("must provide a Dictionary or HashDictionary for gensim engine")
            self.engine = GensimEngine(engine=type, dictionary=dictionary, min_topics=min_topics, max_topics=max_topics, **kwargs)
        elif engine == "sklearn":
            if dictionary is None:
                raise ValueError("must provide a CountVectorizer or HashVectorizer for sklearn engine")
            self.engine = SklearnEngine(engine=type, vectorizer=dictionary, min_topics=min_topics, max_topics=max_topics, **kwargs)
        else:
            raise ValueError("engine must be either 'gensim' or 'sklearn'.")

    def model(self, corpus=None):
        """
        Fit topic models using the current engine. Takes as input a list or iterable of
        documents in bag-of-words format. This enables multiple engines to be used from
        the same API.
        """
        if corpus is None:
            raise ValueError("must provide a list or iterable of BoW documents")
        self.engine.fit(corpus)

    def update(self, documents=None):
        """
        Updates the models with new documents.
        """
        if documents is None:
            raise ValueError("must provide a list or iterable of BoW documents")
        self.engine.update(documents)

    def topics(self):
        """
        Returns the topics for the models.
        """
        return self.engine.topics()