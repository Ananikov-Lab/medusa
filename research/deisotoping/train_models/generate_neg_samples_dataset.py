from typing import List, Optional
from argparse import ArgumentParser
import pickle as pkl
import random

import numpy as np
from tqdm import tqdm

from mass_automation.deisotoping.process import find_spec_peaks
from mass_automation.formula import Formula
from mass_automation.formula.check_formula import del_isotopologues
from research.deisotoping.train_models.generate_mixtures import get_noise_peaks

NEUTRON_MASS = 1.00866501
AVERAGE_MASS = 0.998


def gen_neg_dataset(substances_dictionary):
    # inner parameters
    n_scans = 16
    n_points = 2
    noise_peaks = get_noise_peaks(n_scans, n_points)[1]
    lower_content_threshold = noise_peaks.mean() + noise_peaks.std() * 3
    upper_content_threshold = n_scans * 10 ** 9 * random.choice([0.3, 0.5, 0.7, 1])

    X = np.array(10 * [0]).reshape(1, 10)
    targets = np.array([0])

    for key in tqdm(substances_dictionary.keys()):
        dist_error = 2.6 * 10 ** (-4)

        cmpd1 = Formula(key)
        masses1, ints1 = cmpd1.isodistribution()
        masses1, ints1 = del_isotopologues(masses1, ints1)

        mass1 = masses1[-1]
        mass1 = np.random.uniform(mass1 - dist_error, mass1 + dist_error)

        content1 = np.random.uniform(np.log10(100 * lower_content_threshold), np.log10(upper_content_threshold))
        ints1 = 10 ** content1 * ints1
        int1 = ints1[-1]

        cmpd2 = Formula(substances_dictionary[key])
        masses2, ints2 = cmpd2.isodistribution()
        masses2, ints2 = del_isotopologues(masses2, ints2)

        mass2 = masses2[1]
        mass2 = np.random.uniform(mass2 - dist_error, mass2 + dist_error)

        content2 = np.random.uniform(np.log10(100 * lower_content_threshold), np.log10(upper_content_threshold))
        ints2 = 10 ** content2 * ints2
        int2 = ints2[1]

        # Signal-to-noise ratio of ipeak
        snr_ipeak = int1 / noise_peaks.mean()

        # Signal-to-noise ratio of jpeak
        snr_jpeak = int2 / noise_peaks.mean()

        # Distance between ipeak and jpeak
        dist = abs(mass2 - mass1)

        # Number of peaks of the same class as ipeak
        linm = len(masses1)
        if linm == 1:
            linm = 0

        # Recent intensities belonging to the class ipeak
        past_label_peaks = content1 * ints1[-3:] / int2
        if len(past_label_peaks) < 3:
            past_label_peaks = np.concatenate((np.zeros(3 - len(past_label_peaks)), past_label_peaks),
                                              axis=None)
        past_label_peaks = np.zeros(3)

        # Recent distances between peaks, belonging to the class ipeak
        past_label_masses = masses1[-4:]
        if len(past_label_masses) <= 3:
            past_label_masses = np.concatenate(
                (np.zeros(4 - len(past_label_masses)), past_label_masses),
                axis=None)
        past_label_distances = np.diff(past_label_masses)
        past_label_distances = np.zeros(3)

        # Feature and target arrays generation
        features = np.array([snr_ipeak, snr_jpeak, dist, linm])
        features = np.concatenate((features, past_label_peaks), axis=None)
        features = np.concatenate((features, past_label_distances), axis=None)
        features = features.reshape(1, 10)
        X = np.append(X, features, axis=0)

        target = 0
        targets = np.append(targets, target)

    if X.shape[0] != len(targets):
        print(X.shape, len(targets))
        raise ValueError("The lengths of feature and target arrays are not equal.")

    return X, targets


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('dict_path', help='Path to the pickle, containing substances dictionary')
    parser.add_argument('save_path', help='Path to the generated dataset')
    args = parser.parse_args()
    dict_path = args.dict_path
    save_path = args.save_path

    with open(dict_path, 'rb') as f:
        substances_dictionary = pkl.load(f)

    generated_dataset = gen_neg_dataset(substances_dictionary)
    with open(save_path, 'wb') as f:
        pkl.dump(generated_dataset, f)
