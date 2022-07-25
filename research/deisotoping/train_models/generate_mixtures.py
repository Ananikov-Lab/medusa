import random
from argparse import ArgumentParser
import pickle as pkl
from typing import List, Tuple, Optional
from tqdm import tqdm
import gc

import numpy as np
from pyteomics.mass.mass import isotopologues, isotopic_composition_abundance, calculate_mass

from mass_automation.experiment import Spectrum, Experiment
from mass_automation.formula.check_formula import del_isotopologues

ELECTRON_MASS = 0.00054858

from threading import Thread
import functools


def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]

            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e

            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret

        return wrapper

    return deco


def open_molecules(path=r'D:\files_for_research\0.1_subsample.merged.subsampled_honestly.txt', type='txt'):
    """ The function opens the file, containing PubChem subsample

    Parameters
    ----------
    path : str
        The path to the file (default is "D:\files_for_research\0.1_subsample.merged.subsampled_honestly.txt").
    type : str
        Type of the file

    Returns
    -------
    If type is 'pkl'
    List
    The 2d-array, with the following structure : [[compound_name, <1x200000 sparse matrix of
    type '<class 'numpy.float64'>' with 7 stored elements in Compressed Sparse Row format>], ...].
    If type is 'txt'
    List
    List, which contains substances' names
    """
    if type == 'pkl':
        pickle_file = open(path, 'rb')
        return pkl.load(pickle_file)

    if type == 'txt':
        file_object = open(path, 'r')
        subsample = file_object.read()
        return subsample.split('\n')


def create_substance_list(objects: List, size: Optional[int] = 200, subsampling=True, add_ions: Optional[bool] = True):
    """ Creates the list of substance's names and charges (default is 1), which will be presented in syntetic mixture.

    Parameters
    ----------
    objects : List
        If pkl, the 2d-array, with the following structure : [[compound_name, <1x200000 sparse matrix of type
        '<class 'numpy.float64'>' with 7 stored elements in Compressed Sparse Row format>], ...].
        If txt, the list with compounds' names.
    size : int
        The length of the created substance list.
    subsampling : bool
        If True, subsampling is performed.
    add_ions : bool
        If True, H, Na or K ion is added to the molecule with different probability to make spectra more realistic.
    Returns
    -------
    List[Dict]
    The list, which contains dictionaries with formulas' names and charges.
    """
    if subsampling:
        list_of_indexes = np.random.randint(len(objects), size=size)
    else:
        list_of_indexes = np.arange(size)

    list_of_compounds = []
    for index in list_of_indexes:
        if type(objects) is not list:
            formula = objects[index][0].split('\n')[0]
        else:
            formula = objects[index]

        if add_ions:
            formula = formula + random.choice(10 * ['H'] + ['Na', 'K'])
        list_of_compounds.append({'formula': formula, 'charge': 1})

    return list_of_compounds


def get_noise_peaks(n_scans: Optional[int], n_points: Optional[int]):
    """ Function randomly selects a file with recorded noise taken at a given number of scans
    and number of points (in millions) and performs its randomization.

    Parameters
    ----------
    n_scans : int
        The number of scans.
    n_points : int
        The number of points in millions.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    The tuple with masses and intensities' arrays.
    """
    exp = Experiment(f'D:\\mass_spectra\\Noise_spectra\\ESI-{n_points}m-{n_scans}.mzXML', n_scans, n_points)
    spectrum = exp[0]
    subtracted_mass = spectrum.masses[0]
    masses = spectrum.masses - subtracted_mass
    noise_ints = np.where(spectrum.ints < (np.median(spectrum.ints) + 2 * np.std(spectrum.ints)), spectrum.ints,
                          np.median(spectrum.ints))
    np.random.shuffle(noise_ints)
    del exp, spectrum
    gc.collect()
    return masses, noise_ints


def create_syntetic_mixture(n_of_cmpds: Optional[int], spectrum_options: Tuple[int, int], objects: List,
                            concentrations=True, subsampling=True, add_ions=True):
    """According to the number of compounds, the number of scans and the number of points,
    the synthetic mixture is generated.

    Parameters
    ----------
    n_of_cmpds : int
        The number of compounds.
    spectrum_options : Tuple
        The tuple, where the first element is the number of scans and the second one is the number of points.
    objects : List
        List with compounds' names.
    concentrations : bool
        Imitates different concentrations. (default is True)
    subsampling : bool
        If True, subsampling is performed.
    add_ions : bool
        If True, H, Na or K ion is added to the molecule with different probability to make spectra more realistic.
        (default is True)
    Returns
    -------
    Spectrum.
    """
    n_scans = spectrum_options[0]
    n_points = spectrum_options[1]

    # We are going to append data here
    global_masses = []
    global_peaks = []
    global_labels = []

    # Noise overlay
    gnp = get_noise_peaks(n_scans, n_points)
    noise_masses = gnp[0]
    noise_peaks = gnp[1]
    print('noise overlayed')
    # Let the signal-to-noise ratio be greater than or equal to three
    lower_content_threshold = noise_peaks.mean() + noise_peaks.std() * 3
    upper_content_threshold = n_scans * 10 ** 9 * random.choice([0.3, 0.5, 0.7, 1])

    global_masses.extend(noise_masses)
    global_peaks.extend(noise_peaks)
    label = -1
    global_labels.extend(len(noise_masses) * [label])
    label += 1

    # Peak generation from substances
    for compound in tqdm(create_substance_list(objects, n_of_cmpds, subsampling, add_ions)):

        if compound['charge'] != 1:
            raise NotImplementedError

        # calibration error in the masses so models don't over-learn
        cal_error = np.random.uniform(-(2 * 10 ** -3), (2 * 10 ** -3))

        def get_isotopes(formula, charge):
            isotopes = [[], []]
            for isotope in isotopologues(formula, charge, overall_threshold=0.0001):
                isotopes[0].append(calculate_mass(isotope) - charge * ELECTRON_MASS - cal_error)
                isotopes[1].append(isotopic_composition_abundance(isotope))
            return isotopes

        func = timeout(timeout=30)(get_isotopes)
        try:
            isotopes = func(compound['formula'], compound['charge'])
        except:
            isotopes = [[], []]

        if len(isotopes[0]) != 0:

            isotopes[0] = np.array(isotopes[0])
            # distance error in the masses so models don't over-learn
            dist_error = 2.6 * 10 ** (-4)
            for i in range(len(isotopes[0])):
                isotopes[0][i] = np.random.uniform(isotopes[0][i] - dist_error, isotopes[0][i] + dist_error)

            isotopes[1] = np.array(isotopes[1])

            isotopes[1] = isotopes[1][np.argsort(isotopes[0])]
            isotopes[0] = np.sort(isotopes[0])

            side_threshold = 0.001
            greater_than_thresh = np.where(isotopes[1] > side_threshold)[0]

            pos_first = greater_than_thresh[0]
            pos_last = greater_than_thresh[-1]

            isotopes[1] = isotopes[1][pos_first:pos_last + 1]
            isotopes[0] = isotopes[0][pos_first:pos_last + 1]

            masses = isotopes[0]
            peaks = isotopes[1] / isotopes[1].max()

            if concentrations:
                content = np.random.uniform(np.log10(100 * lower_content_threshold), np.log10(upper_content_threshold))
            else:
                content = np.log10(upper_content_threshold)

            peaks = 10 ** content * peaks
            del_indexes = np.where(peaks <= lower_content_threshold)[0]

            masses = np.delete(masses, del_indexes)
            peaks = np.delete(peaks, del_indexes)

            # Delete the isotopologues
            corr_masses, corr_peaks = del_isotopologues(masses, peaks)
            del_masses = masses[np.in1d(masses, corr_masses) != True]
            del_peaks = peaks[np.in1d(peaks, corr_peaks) != True]

            global_masses.extend(corr_masses)
            global_peaks.extend(corr_peaks)
            global_labels.extend(len(corr_masses) * [label])

            global_masses.extend(del_masses)
            global_peaks.extend(del_peaks)
            global_labels.extend(len(del_masses) * [-1])

            label += 1
    print('compounds added')

    # Sorting
    global_masses = np.array(global_masses)
    global_peaks = np.array(global_peaks)
    global_labels = np.array(global_labels)

    global_peaks = global_peaks[np.argsort(global_masses)]
    global_labels = global_labels[np.argsort(global_masses)]
    global_masses = np.sort(global_masses)

    spectrum = Spectrum(global_masses, global_peaks, n_scans, n_points, path='')
    return spectrum, global_labels


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('n_mixtures', type=int, help="Number of mixtures")
    parser.add_argument('n_compounds_lower_limit', type=int, help="Lower limit number of compounds in the mixture")
    parser.add_argument('n_compounds_upper_limit', type=int, help="Upper limit number of compounds in the mixture")
    parser.add_argument('n_scans', type=int, help="Number of scans in spectra. If 0, random number of scans given")
    parser.add_argument('n_points', type=int,
                        help="Number of millions of points in spectra. If 0, random number of million points given")
    parser.add_argument('concentrations', type=bool, help="Add concentrations or not")
    parser.add_argument('subsampling', type=bool, help="Do subsampling or not")
    parser.add_argument('add_ions', type=bool, help="Add ions or not")
    parser.add_argument('path', type=str, help='The path, where lists are saved')

    args = parser.parse_args()

    n_mixtures = args.n_mixtures
    low_lim = args.n_compounds_lower_limit
    up_lim = args.n_compounds_upper_limit
    n_scans = args.n_scans
    n_points = args.n_points
    concentrations = args.concentrations
    subsampling = args.subsampling
    add_ions = args.add_ions
    path = args.path

    list_of_spectrum_options = [(8, 2), (16, 2), (32, 2), (64, 2), (128, 2),
                                (8, 4), (16, 4), (32, 4), (64, 4), (8, 8),
                                (16, 8), (8, 16), (16, 16)]

    # The creation of the list with the syntetic mixtures for dumping
    syntetic_mixtures = []
    for i in range(n_mixtures):
        objects = open_molecules()
        spectrum_options = random.choice(list_of_spectrum_options)
        n_of_cmpds = random.randint(low_lim, up_lim)
        if (n_scans != 0) and (n_points != 0):
            spectrum_options = (n_scans, n_points)
        try:
            csm = create_syntetic_mixture(n_of_cmpds, spectrum_options, objects=objects, concentrations=concentrations,
                                          subsampling=subsampling, add_ions=add_ions)
        except:
            csm = create_syntetic_mixture(n_of_cmpds, spectrum_options, objects=objects, concentrations=concentrations,
                                          subsampling=subsampling, add_ions=add_ions)
        syntetic_mixtures.append(csm)
        print(i)

    with open(path, 'wb') as f:
        pkl.dump(syntetic_mixtures, f)
