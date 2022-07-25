import numpy as np

EPS = 1e-5


def compute_entropy(p):
    return -p * np.log2(p + EPS) - (1 - p) * np.log2(1 - p + EPS)


def compute_mean_entropy(ps):
    return np.mean([compute_entropy(p) for p in ps], axis=0)


class BinaryProbaModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, *args, **kwargs):
        return self.model.predict_proba(*args, **kwargs)[:, 1]


class EnsembleWrapper:
    """Wrapper for an ensemble of models.
    """

    def __init__(self, model_type, models):
        if model_type not in ['clf', 'reg']:
            raise ValueError('type must be either clf or reg')

        self.type = model_type
        self.models = models

    def predict_all(self, *args, **kwargs):
        predictions = []

        for model in self.models:
            predictions.append(model.predict(*args, **kwargs))

        return predictions

    def predict(self, *args, **kwargs):
        return np.mean(self.predict_all(*args, **kwargs), axis=0)

    def predict_w_uncertainty(self, *args, **kwargs):
        predictions = self.predict_all(*args, **kwargs)
        prediction = np.mean(predictions, axis=0)

        if self.type == 'clf':
            total_uncertainty = compute_entropy(prediction)
            data_uncertainty = compute_mean_entropy(predictions)

            knowledge_uncertainty = total_uncertainty - data_uncertainty

            return prediction, data_uncertainty, knowledge_uncertainty, total_uncertainty

        elif self.type == 'reg':
            knowledge_uncertainty = np.var(predictions, axis=0)

            return prediction, knowledge_uncertainty
