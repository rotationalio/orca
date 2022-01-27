class ModelingEngine():
    """
    ModelingEngine defines a common interface for performing text modeling across
    different modeling APIs.
    """

    def __init__(self):
        self.models = {}

    def fit(self, model, dictionary, corpus):
        """
        Fits a set of models on the provided corpus.
        """
        raise NotImplementedError("fit() must be implemented by a subclass.")

    def update(self, documents):
        """
        Updates the set of models with a batch or stream of new corpus documents.
        """
        raise NotImplementedError("update() must be implemented by a subclass.")

    def topics(self):
        """
        Return the evaluation results and metrics for the set of current models.
        """
        raise NotImplementedError("results() must be implemented by a subclass.")