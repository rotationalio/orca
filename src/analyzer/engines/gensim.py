from src.analyzer.engines.engine import ModelingEngine

from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Nmf

class GensimEngine(ModelingEngine):
    """
    GensimEngine implements the methods for fitting and evaluating gensim models on a
    corpus.
    """
    def __init__(self, min_topics, max_topics):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.models = {}
        self.dictionary = None
        self.type = ""
        self.corpus = None

    def fit(self, model, dictionary, corpus, **kwargs):
        """
        Fits a set of models on the provided corpus.
        """
        if model == "lda":
            self._fit_models(LdaModel, dictionary, corpus, **kwargs)
        elif model == "nmf":
            tfidf = TfidfModel(dictionary=dictionary)
            self._fit_models(Nmf, dictionary, tfidf[corpus], **kwargs)
        else:
            raise ValueError("model must be either lda or nmf.")
        self.dictionary = dictionary
        self.type = model
        self.corpus = corpus

    def _fit_models(self, model_class, dictionary, corpus, **kwargs):
        """
        Fits several models with different numbers of topics on the given corpus.
        """
        for num_topics in range(self.min_topics, self.max_topics + 1):
            print("fitting {} model with {} topics".format(model_class.__name__, num_topics))
            model = model_class(corpus=corpus, id2word=dictionary, num_topics=num_topics, **kwargs)
            cm = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            results = {}
            results['coherence'] = cm.get_coherence()
            print("coherence: {}".format(results['coherence']))
            self.models[num_topics] = {}
            self.models[num_topics]['model'] = model
            self.models[num_topics]['results'] = results

    def update(self, documents):
        """
        Updates the set of models with a stream of new corpus documents.
        """
        bow = [self.dictionary.doc2bow(doc, allow_update=True) for doc in documents]
        self.corpus.append(bow)
        for _, m in self.models.items():
            m['model'].update(bow)
            cm = CoherenceModel(model=m['model'], corpus=self.corpus, dictionary=self.dictionary, coherence='u_mass')
            m['results']['coherence'] = cm.get_coherence()

    def topics(self):
        """
        Return the discovered topics and evaluation metrics for the set of current models.
        """
        topics = {}
        for k, m in self.models.items():
            topics[k] = {}
            topics[k]['coherence'] = m['results']['coherence']
            topics[k]['topics'] = m['model'].show_topics(num_topics=k)
        return topics