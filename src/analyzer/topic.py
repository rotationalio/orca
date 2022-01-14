from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.tfidfmodel import TfidfModel
from gensim.models import Nmf

class TopicAnalyzer():
    """
    TopicAnalyzer supports fitting various topic models on a gensim corpus.
    """
    def __init__(self, min_topics=5, max_topics=10):
        self.min_topics = min_topics
        self.max_topics = max_topics
        self.models = {}

    def _fit_models(self, model_class, dictionary, corpus, **kwargs):
        """
        Fits several models of the given type on the given dictionary and corpus with the given set of parameters.
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

    def fit_lda(self, dictionary, corpus, **kwargs):
        """
        Fits several LDA models on the given dictionary and corpus with the given set of
        parameters.
        """
        self._fit_models(LdaModel, dictionary, corpus, **kwargs)
        
    def fit_nmf(self, dictionary, corpus, **kwargs):
        """
        Fits several NMF models on the given dictionary and corpus with the given set of
        parameters.
        """
        tfidf = TfidfModel(dictionary=dictionary)
        self._fit_models(Nmf, dictionary, tfidf[corpus], **kwargs)

    def update(self, documents):
        """
        Updates the LDA model with a stream of new corpus documents.
        """
        for m in self.models:
            model = m['model'].update(documents)

    def topics(self):
        """
        Returns the topics for the models.
        """
        topics = {}
        for k, m in self.models.items():
            topics[k] = {}
            topics[k]['coherence'] = m['results']['coherence']
            topics[k]['topics'] = m['model'].show_topics(num_topics=k)
        return topics
