import pickle as pkl
from abc import ABC, abstractmethod
from typing import Optional, List
from tqdm.autonotebook import tqdm

import numpy as np
from ..experiment import Spectrum

NEUTRON_MASS = 1.00866501
AVERAGE_MASS = 0.998


def find_spec_peaks(spectrum: Spectrum, min_distance: float, algorithm='std', verbose=False, alpha=None,
                    threshold=None):
    """ Peak finding algorithm

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum, where peaks are searched.
    min_distance : float
        Required minimal mass difference between neighbouring peaks.
    algorithm : {‘std’, ‘hist’, ‘quantile’, 'manual'}, default=’std’
        If 'std', minimal intensity is set as spectra ints median + double standard
        deviation.
        If 'hist', minimal intensity peak threshold is set as the first bin edge in spectra
        of intensities.
        If 'quantile', minimal intensity is set as parameter alpha quantile.
        If None, minimal intensity is set manually.
    verbose : bool
            If True, show TQDM bars.
    alpha : float       
        Argument in quantile function. Used in 'quantile' algorithm. 
    threshold : float
        Minimal peak intensity threshold value in 'manual' algorithm.

    Returns
    -------
    List[np.array, np.array, np.array]
        List with three arrays (peak masses, peak intensities, peak labels)

    """
    if threshold is None:
        if algorithm is None:
            raise ValueError('Set algorithm or minimal intensity threshold value')
        elif algorithm == 'std':
            threshold = np.median(spectrum.ints) + 2 * np.std(spectrum.ints)
        elif algorithm == 'hist':
            hist, bin_edges = np.histogram(spectrum.ints, bins=int(len(spectrum.ints) / 100))
            threshold = bin_edges[1]
        elif algorithm == 'quantile':
            if alpha:
                threshold = np.quantile(spectrum.ints, alpha)
            else:
                threshold = np.quantile(spectrum.ints, 0.995)
        else:
            raise ValueError('Unknown algorithm')

    cand_peaks = [[], [], []]
    zero_out = np.where(spectrum.ints > threshold, spectrum.ints, 0)
    cand_peaks[0] = spectrum.masses[zero_out != 0]
    cand_peaks[1] = spectrum.ints[zero_out != 0]
    cand_peaks[2] = np.arange(len(spectrum.masses))[zero_out != 0]

    peaks_to_be_selected = [[], [], []]

    for mass, int_, i in tqdm(zip(*cand_peaks), total=len(cand_peaks[0]), disable=(not verbose)):
        mass_low = cand_peaks[0] > mass - min_distance
        mass_high = cand_peaks[0] < mass + min_distance

        if cand_peaks[1][mass_low & mass_high].max() == int_:
            if (i != 0) and (i != (len(spectrum.ints) - 1)):
                if (spectrum.ints[i - 1] < spectrum.ints[i]) and (spectrum.ints[i] > spectrum.ints[i + 1]):
                    peaks_to_be_selected[0].append(mass)
                    peaks_to_be_selected[1].append(int_)
                    peaks_to_be_selected[2].append(i)

    peaks_to_be_selected = [np.array(peaks_to_be_selected[0]),
                            np.array(peaks_to_be_selected[1]),
                            np.array(peaks_to_be_selected[2])]

    return peaks_to_be_selected


class FeatureLogger:
    """Class for checking deisotoping model predictions in spectra

    Attributes
    ----------
    features : List
        List containing counted features for each pair of possible peaks in spectrum.
    targets : List
        List containing model answers for each pair of possible peaks in spectrum
    """

    def __init__(self):
        self.features = []
        self.targets = []

    def write(self, features, prediction):
        """ Write features and model answer for new pair of possible peaks in spectrum.

        Parameters
        ----------
        features : np.ndarray
            New pair of possible peaks features
        prediction : np.ndarray
            New pair of possible peaks prediction

        """
        self.features.append(features)
        self.targets.append(prediction)

    def dumpf(self, path):
        """ Dump feature logger dataset

        Parameters
        ----------
        path : str
            Path to the saved pkl-file

        """
        with open(path, 'rb') as f:
            pkl.dump(self, f)


class Deisotoper(ABC):
    """General class for deisotoping algorithms.

    Attributes
    ----------
    model : dict
        The dictionary, which contains model parameters and ML model, if necessary.
    """

    def __init__(self, model=None):
        """
        Parameters
        ----------
        model : dict
            The dictionary, which contains model parameters and ML model, if necessary.
            Dictionary structure:
            { "min_distance" : 0.021,
              "delta" : 0.007,
              "model" : Machine learning model object }
        """
        self.model = model
        self.denoised_labels = np.array([])
        self.threshold = 0

    def find_peaks(self, spectrum: Spectrum) -> np.ndarray:
        """ Peak finding implementation.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum, which should be processed by the peak finding algorithm.

        Returns
        -------
        np.ndarray
            The array of peaks' indices.
        """
        min_dist = self.model['min_distance']
        peaks = find_spec_peaks(spectrum, min_distance=min_dist, algorithm='std')[2]
        denoised_labels = np.zeros(len(spectrum.masses)) - 1
        denoised_labels[peaks] = 0
        self.denoised_labels = denoised_labels
        return peaks

    def load(self, path=None, min_distance=0.01, delta=0.007):
        """ Loading the model.

        Parameters
        ----------
        path : str
            The path to the pkl file, which contains the model dictionary.
        min_distance : float
            The minimal mass difference between neighbouring peaks.
        delta : float
            The mass difference between neighbouring peaks.
        """
        if path is None:
            self.model = {
                'min_distance': min_distance,
                'delta': delta
            }
        else:
            if self.model is None:
                with open(path, 'rb') as f:
                    self.model = pkl.load(f)
            else:
                raise ValueError("Already loaded")

        return self

    def run(self, spectrum: Spectrum, threshold=0.5, logger=None,
            count_features=13, exp_coef=1) -> np.ndarray:
        """
        The deisotoping algorithm. Returns the array of labels, assigns labels
        to spectrum.isotopic_distributions and assigns True to spectrum.deisotoped.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum, which should be deisotoped.
        threshold : float
            Minimal predict proba of ML algorithm. (default is 0.5)
        logger : FeatureLogger object
            Logger for deisotoping process in spectrum
        count_features : int
            Number of features in deisotoping model. (default is 13)
        exp_coef:
            Coefficient of proportionality between delta parameter and radius of vicinity,
            where possible peaks can be located. (default is 1)

        Returns
        -------
        np.ndarray
            The array of labels.
        """

        masses = spectrum.masses
        peaks = self.find_peaks(spectrum)

        delta = self.model['delta']

        labels = np.zeros(len(masses)) - 1

        counter = 0
        for i, ipeak in enumerate(tqdm(peaks[:-1])):
            preds = []
            current_peaks = []
            for j, jpeak in enumerate(peaks[i + 1:]):
                # check for anomalous mass difference
                if abs(masses[jpeak] - masses[ipeak] - AVERAGE_MASS) < exp_coef * delta:
                    features = self.calc_features(spectrum, ipeak, jpeak, labels, count_features)

                    pred = self.make_prediction(features)

                    if logger:
                        logger.write(features, pred)

                    preds.append(pred)
                    current_peaks.append(jpeak)

                if abs(masses[jpeak] - masses[ipeak]) > AVERAGE_MASS + exp_coef * delta:
                    break

            if len(preds) != 0:
                if self.__class__ is LinearDeisotoper:
                    min_idx = np.argmin(preds)
                    jpeak = current_peaks[min_idx]

                    if preds[min_idx] < delta:
                        if labels[ipeak] == -1:
                            labels[ipeak] = counter
                            labels[jpeak] = counter
                            counter += 1
                        else:
                            labels[jpeak] = labels[ipeak]

                if self.__class__ is MlDeisotoper:
                    max_idx = np.argmax(preds)
                    jpeak = current_peaks[max_idx]

                    if preds[max_idx] > threshold:
                        if labels[ipeak] == -1:
                            labels[ipeak] = counter
                            labels[jpeak] = counter
                            counter += 1
                        else:
                            labels[jpeak] = labels[ipeak]

        spectrum.deisotoped = True
        spectrum.isotopic_distributions = labels
        self.threshold = threshold
        return labels

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abstractmethod
    def calc_features(self, spectrum: Spectrum, ipeak: int, jpeak: int,
                      labels: np.ndarray, count_features: int) -> np.ndarray:
        """Calculates features for an object, which is a pair of peaks.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum, where we select the peaks.
        ipeak : int
            The indice of the first peak in the pair.
        jpeak : int
            The index of the second peak in the pair
        labels : np.ndarray
            The array of labels in the state before the features calculation.
        count_features : int
            The number of features in the model.

        Returns
        -------
        np.ndarray
            The array of calculated features.

            If LinearDeisotoper:

                array = [masses[ipeak], masses[jpeak]]

            If MlDeisotoper:

                array = ["Distance between ipeak and jpeak",
                         "Number of peaks, registered before classification",
                         "Recent intensities belonging to the class ipeak (5 numbers)",
                         "Recent distances belonging to the class ipeak (3 numbers)",
                         "Maximal intensity of peaks labeled as ipeak to jpeak intensity relation",
                         "Mean intensity of peaks labeled as ipeak to jpeak intensity relation",
                         "Mean intensity of last 3 labeled as ipeak peaks compared to max intensity"]
        """
        pass

    @abstractmethod
    def make_prediction(self, features) -> bool:
        """Performs classifier prediction.

        Parameters
        ----------
        features : np.ndarray
            The array of features for the pair of peaks

        Returns
        -------
        bool
            If True, the peaks are in the same class, else not.
        """
        pass


class LinearDeisotoper(Deisotoper):
    def calc_features(self, spectrum: Spectrum, ipeak: int, jpeak: int,
                      labels: Optional[np.ndarray] = None, count_features=13) -> np.ndarray:
        return np.array([spectrum.masses[ipeak], spectrum.masses[jpeak]])

    def make_prediction(self, features: np.ndarray) -> bool:
        return abs(features[1] - features[0] - AVERAGE_MASS)


class MlDeisotoper(Deisotoper):
    def calc_features(self, spectrum: Spectrum, ipeak: int, jpeak: int, labels: np.ndarray,
                      count_features=13) -> np.ndarray:
        masses = spectrum.masses
        ints = spectrum.ints

        # 1) Distance between ipeak and jpeak
        dist = abs(masses[jpeak] - masses[ipeak])

        # 2) Number of peaks, labeled as ipeak
        if labels[ipeak] != -1:
            numb = np.count_nonzero(labels[:ipeak + 1] == labels[ipeak])
            if numb == 1:
                numb = 0
        else:
            numb = 0

        # (3-7) Recent 5 intensities belonging to the class ipeak
        if labels[ipeak] != -1:
            past_label_peaks = ints[:ipeak + 1][labels[:ipeak + 1] == labels[ipeak]][-5:]
            if len(past_label_peaks) < 5:
                past_label_peaks = np.concatenate((np.zeros(5 - len(past_label_peaks)), past_label_peaks),
                                                  axis=None)
            past_label_peaks = past_label_peaks / ints[jpeak]
        else:
            past_label_peaks = np.array([0, 0, 0, 0, ints[ipeak] / ints[jpeak]])

        # (8-10) Recent 3 distances between peaks, belonging to the class ipeak
        if labels[ipeak] != -1:
            past_label_masses = masses[:ipeak + 1][labels[:ipeak + 1] == labels[ipeak]][-4:]
            if len(past_label_masses) <= 3:
                past_label_masses = np.concatenate(
                    (np.zeros(4 - len(past_label_masses)), past_label_masses),
                    axis=None)
            past_label_distances = np.diff(past_label_masses)
        else:
            past_label_distances = np.zeros(3)

        past_label_distances = np.where(past_label_distances > 2, 0, past_label_distances)

        # (11-13) maximal and mean intensity of peaks labeled as ipeak to jpeak intensity relation
        # + mean intensity of last 3 labeled as ipeak peaks compared to max intensity

        if labels[ipeak] != -1:
            labeled_as_ipeak = ints[:ipeak + 1][labels[:ipeak + 1] == labels[ipeak]]
            max_to_j = labeled_as_ipeak.max() / ints[jpeak]
            mean_to_j = labeled_as_ipeak.mean() / ints[jpeak]
            past_3_peaks = past_label_peaks[-3:] * ints[jpeak]
            mean_3_to_max = past_3_peaks[past_3_peaks != 0].mean() / labeled_as_ipeak.max()
        else:
            max_to_j = ints[ipeak] / ints[jpeak]
            mean_to_j = ints[ipeak] / ints[jpeak]
            mean_3_to_max = 1

        # Feature array generation
        features = np.array([dist, numb])
        features = np.concatenate((features, past_label_peaks), axis=None)
        features = np.concatenate((features, past_label_distances), axis=None)
        extra_features = np.array([max_to_j, mean_to_j, mean_3_to_max])
        features = np.concatenate((features, extra_features), axis=None)
        features = features.reshape(1, count_features)
        return features

    def make_prediction(self, features: np.ndarray) -> bool:
        clf = self.model['model']
        pred = clf.predict_proba(features)[0][1]
        return pred
