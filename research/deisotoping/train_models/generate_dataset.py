from typing import List, Optional
from argparse import ArgumentParser
import pickle as pkl

import numpy as np
from tqdm import tqdm

from mass_automation.deisotoping.process import find_spec_peaks

NEUTRON_MASS = 1.00866501
AVERAGE_MASS = 0.998


def gen_dataset(list_of_csms: List, delta: Optional[float] = 0.007,
                min_distance: Optional[float] = 0.01, count_features=13):
    """ Uses a pack with artificial mass-spectra and generates training dataset.

    Parameters
    ----------
    list_of_csms : List
        The list with the tuples, which identify artificial mass spectra (from "create_syntetic_mixture" function).

    delta : float
        The parameter in linear model. If the distance between peaks is more than 10*delta, these objects are not
        added to the training dataset (default is 0.007).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    The tuple with 2 numpy arrays. The first one is a 2D array where rows are the objects (the pairs of peaks),
    cols are the features. The second array is the array of targets (1s or 0s).
    """
    X = np.array(count_features * [0]).reshape(1, count_features)
    targets = np.array([0])
    for csm in tqdm(list_of_csms):
        spectrum, labels = csm
        masses = spectrum.masses
        ints = spectrum.ints
        peaks = find_spec_peaks(spectrum, min_distance=min_distance)[2]

        for i, ipeak in enumerate(peaks[:-1]):
            for j, jpeak in enumerate(peaks[i + 1:]):
                # Checking for anomalous mass difference
                if abs(masses[jpeak] - masses[ipeak] - AVERAGE_MASS) < 30 * delta:

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
                        past_label_peaks = np.zeros(5)

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

                    """
                    (11-13) maximal and mean intensity of peaks labeled as ipeak to jpeak intensity relation
                    + mean intensity of last 3 labeled as ipeak peaks compared to max intensity
                    """
                    if labels[ipeak] != -1:
                        labeled_as_ipeak = ints[:ipeak + 1][labels[:ipeak + 1] == labels[ipeak]]
                        max_to_j = labeled_as_ipeak.max() / ints[jpeak]
                        mean_to_j = labeled_as_ipeak.mean() / ints[jpeak]
                        past_3_peaks = past_label_peaks[-3:] * ints[jpeak]
                        mean_3_to_max = past_3_peaks[past_3_peaks != 0].mean() / labeled_as_ipeak.max()
                    else:
                        max_to_j = 0
                        mean_to_j = 0
                        mean_3_to_max = 1

                    # Feature and target arrays generation
                    features = np.array([dist, numb])
                    features = np.concatenate((features, past_label_peaks), axis=None)
                    features = np.concatenate((features, past_label_distances), axis=None)
                    extra_features = np.array([max_to_j, mean_to_j, mean_3_to_max])
                    features = np.concatenate((features, extra_features), axis=None)
                    features = features.reshape(1, count_features)
                    X = np.append(X, features, axis=0)

                    target = (labels[ipeak] == labels[jpeak] and labels[ipeak] != -1)
                    targets = np.append(targets, target)

    if X.shape[0] != len(targets):
        print(X.shape, len(targets))
        raise ValueError("The lengths of feature and target arrays are not equal.")

    return X, targets


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('delta', type=float, help="The parameter in model. If the distance between peaks is more"
                                                  " than 30*delta, these objects are not added to the training dataset"
                                                  " (default is 0.007).")
    parser.add_argument('min_distance', type=float, help="The parameter in model."
                                                         " If the distance between neighbouring peaks is more"
                                                         " than min_distance, the most intensive only remains")
    parser.add_argument('list_of_csms_path', type=str, help="The path to the pickle file, which contains the list of"
                                                            " synthetic mixtures.")
    parser.add_argument('gen_ds_path', type=str, help="The path to the pkl file, which contains dataset.")

    args = parser.parse_args()
    delta = args.delta
    min_distance = args.min_distance
    list_of_csms_path = args.list_of_csms_path
    gen_ds_path = args.gen_ds_path

    with open(list_of_csms_path, 'rb') as f:
        list_of_csms = pkl.load(f)

    generated_dataset = gen_dataset(list_of_csms, delta, min_distance)
    with open(gen_ds_path, 'wb') as f:
        pkl.dump(generated_dataset, f)
