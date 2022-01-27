from src.analyzer.engines.gensim import GensimEngine
from src.analyzer.engines.sklearn import SklearnEngine

class TopicAnalyzer():
    """
    TopicAnalyzer supports fitting various topic models on a gensim corpus.
    """
    def __init__(self, engine, min_topics=5, max_topics=10):
        if engine == "gensim":
            self.engine = GensimEngine(min_topics, max_topics)
        elif engine == "sklearn":
            self.engine = SklearnEngine(min_topics, max_topics)
        else:
            raise ValueError("engine must be either 'gensim' or 'sklearn'.")

    def model(self, type, dictionary, corpus, **kwargs):
        """
        Fit topic models using the current engine.
        """
        self.engine.fit(type, dictionary, corpus, **kwargs)

    def update(self, documents):
        """
        Updates the models with new documents.
        """
        self.engine.update(documents)

    def topics(self):
        """
        Returns the topics for the models.
        """
        return self.engine.topics()