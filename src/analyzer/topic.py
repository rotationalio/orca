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

    def fit_lda(self, dictionary, corpus, **kwargs):
        """
        Fits several LDA models on the given dictionary and corpus with the given set of
        parameters.
        """
        for num_topics in range(self.min_topics, self.max_topics + 1):
            print("fitting LDA model with {} topics".format(num_topics))
            model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, **kwargs)
            cm = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            results = {}
            results['coherence'] = cm.get_coherence()
            print("coherence: {}".format(results['coherence']))
            self.models[num_topics] = {}
            self.models[num_topics]['model'] = model
            self.models[num_topics]['results'] = results
        
    def fit_nmf(self, dictionary, corpus, **kwargs):
        """
        Fits several NMF models on the given dictionary and corpus with the given set of
        parameters.
        """
        for num_topics in range(self.min_topics, self.max_topics + 1):
            print("fitting NMF model with {} topics".format(num_topics))
            tfidf = TfidfModel(dictionary=dictionary)
            model = Nmf(tfidf[corpus], id2word=dictionary, num_topics=num_topics, **kwargs)
            cm = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
            results = {}
            results['coherence'] = cm.get_coherence()
            print("coherence: {}".format(results['coherence']))
            self.models[num_topics] = {}
            self.models[num_topics]['model'] = model
            self.models[num_topics]['results'] = results

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
