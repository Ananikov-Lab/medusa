import numpy as np


class Shift:
    """Shifts subspectra; shift sampled uniformly from delta
    """

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, vector):
        shift = np.random.randint(*self.delta)

        if shift >= 0:
            return np.append(vector[shift:], np.zeros(shift))
        else:
            return np.append(np.zeros(np.abs(shift)), vector[:shift])


class Add:
    """Adds uniformly sampled intensity shift
    """

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, vector):
        return np.random.uniform(*self.delta) + vector


class RandomNoise:
    """Adds random noise to the subspectrum
    """

    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, vector):
        return self.sigma * np.random.randn(len(vector)) + vector


class Scale:
    """Scales the subspectrum randomly
    """

    def __init__(self, delta):
        self.delta = delta

    def __call__(self, vector):
        return np.random.uniform(*self.delta) * vector


class DroppingAug:
    """Drops a subspectrum with certain probability
    """

    def __init__(self, threshold, probability):
        self.threshold = threshold
        self.probability = probability

    def __call__(self, vector):
        if vector.max() < self.threshold:
            if np.random.random() < self.probability:
                return None
            else:
                return vector
        else:
            return vector


class AugmentationWrapper:
    """Wraps the set of augmentations
    """

    def __init__(self, *args):
        self.augs = args

    def __call__(self, vector):
        out = vector

        for aug in self.augs:
            out = aug(out)

        return out
