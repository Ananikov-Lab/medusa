from typing import Tuple, Union, List, Optional

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine

from ..experiment import Spectrum
from . import Formula

ELECTRON_MASS = 0.00054858


def del_isotopologues(masses: np.ndarray, peaks: np.ndarray, return_ids: Optional[bool] = False) -> Union[
    Tuple[np.ndarray, np.ndarray], List[int]]:
    """Delete low intense isotopologues.

    Parameters
    ----------
    masses : np.ndarray
        The array of masses from the isotopic distribution.
    peaks : np.ndarray
        The array of intensities from the isotopic distribution.
    return_ids : bool
        Return ids of the most intensive peaks
    Returns
    -------
    Union[Tuple[np.ndarray, np.ndarray], List[int]]
         Contains two numpy arrays with new masses and intensities without the isotopologues or just the ids,
         depending on the ``return_ids`` parameter.
    """
    new_masses = np.array([])
    new_peaks = np.array([])

    mass_dict = {}
    peaks_dict = {}
    counter = 0
    mass_dict[counter] = [masses[0]]
    peaks_dict[counter] = [peaks[0]]
    for i in range(1, len(masses)):
        if (masses[i] - masses[i - 1]) < 0.15:
            mass_dict[counter].append(masses[i])
            peaks_dict[counter].append(peaks[i])
        else:
            counter += 1
            mass_dict[counter] = [masses[i]]
            peaks_dict[counter] = [peaks[i]]

    ids = []

    for counter in mass_dict.keys():
        marked_peaks = np.array(peaks_dict[counter])
        index = np.where(peaks == marked_peaks.max())[0][0]
        ids.append(index)

        new_mass = masses[index]
        new_peak = peaks[index]
        new_masses = np.append(new_masses, new_mass)
        new_peaks = np.append(new_peaks, new_peak)

    if return_ids:
        return ids
    else:
        return new_masses, new_peaks


def get_peak_candidates(spectrum, peak_mass, error, distance=1):
    """ Peak finding around the given mass.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum, where function tries to find peaks.
    peak_mass : float
        The given mass, the center of the slice, where function finds peaks
    error : float
        The radius of the slice, where function performs.
    distance : int
        The parameter, used in peak finding algorithm. (default is 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, bool]
    The first element in the tuple is the array of peak masses, the second is the array of peak intensities and
    the third is the array of peak indices in the slice array. If function finds no peaks, it returns the value of
    the given mass, the median intensity in the slice and the array of indices is numpy.array([0]) Fourth element
    indicates, if function finds at least one peak in the spectrum, else not.
    """
    slice_condition = (spectrum.masses >= peak_mass - error) & (spectrum.masses <= peak_mass + error)
    peak_slice_masses = spectrum.masses[slice_condition]
    peak_slice_ints = spectrum.ints[slice_condition]

    pos_peak_indices, _ = find_peaks(peak_slice_ints, height=np.quantile(spectrum.ints, 0.95),
                                     distance=distance)
    pos_peak_masses = peak_slice_masses[pos_peak_indices]
    pos_peak_ints = peak_slice_ints[pos_peak_indices]

    if len(pos_peak_ints) == 0:
        pos_peak_masses = np.array([peak_mass])
        pos_peak_ints = np.array([np.median(peak_slice_ints)])
        pos_peak_indices = np.array([0])
        is_matched = 0
    else:
        is_matched = 1
    return pos_peak_masses, pos_peak_ints, pos_peak_indices, is_matched


def best_peak(pos_peak_masses, pos_peak_ints, teor_vector, ints_vector):
    """ The function chooses the best peak to add by counting the cosine distance between the theoretical vector
    of intensities and the vector of intensities with the registered peak.

    Parameters
    ----------
    pos_peak_masses : np.array
        The array of possible best peak masses.
    pos_peak_ints : np.array
        The array of possible best peak intensities.
    teor_vector : np.array
        The theoretical vector of intensities, which function uses to count the cosine distance.
    ints_vector : np.array
        The vector of previous intensities. By adding possible best peak intensity to the ints_vector,
        the function gets the new possible intensities vector and counts the cosine distance between it and the
        theoretical vector.

    Returns
    -------
    Tuple[float, float]
    The tuple contains the best peak's mass and the best peak's intensity.
    """
    metrics_dict = {}
    for mass, int_ in zip(pos_peak_masses, pos_peak_ints):
        metrics_dict[(mass, int_)] = cosine(teor_vector, np.append(ints_vector, int_))
    best_peak_mass, best_peak_int = min(metrics_dict, key=metrics_dict.get)
    return best_peak_mass, best_peak_int


def add_new_peak(spectrum, delta, vector, teor_vector, dist_error, distance=1):
    """ Adds best peak's mass and intensity to the array of registered masses and intensities respectively.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum, where function gets peak candidates and chooses the best one to add to the
        resulting vector.
    delta : float
        The distance between the last mass in the vector of masses and the center of the slice, where
        get_peak_candidates tries to find peaks.
    vector : np.ndarray
        The 2D-array, where the first array is the vector of masses, where the function should add the best
        mass and the second array id the array of intensities, where the function should add the best intensity.
    teor_vector : np.array
        The theoretical vector of intensities, which best_peak uses to count the cosine distance.
    dist_error : float
        The possible error in distance between the peaks, characterizes the radius of the second and the next
        peak's vicinity.
    distance : int
        The parameter, used in peak finding algorithm. (default is 1).

    Returns
    -------
    np.ndarray
    The 2D numpy array of masses and intensities with the best chosen last mass and intensity respectively
    and indicator of matching the peak.
    """
    masses_vector = vector[0]
    ints_vector = vector[1]
    pos_peak_masses, pos_peak_ints, pos_peak_indices, is_matched = get_peak_candidates(spectrum,
                                                                                       masses_vector[-1] + delta,
                                                                                       dist_error,
                                                                                       distance=distance)
    best_peak_mass, best_peak_int = best_peak(pos_peak_masses, pos_peak_ints, teor_vector, ints_vector)
    add_vector = np.array([[best_peak_mass], [best_peak_int]])
    return np.hstack((vector, add_vector)), is_matched


def check_presence(spectrum, formula, cal_error=0.006, dist_error=0.003, distance=50, max_peaks=5):
    """
    Calculates the cosine distance between the peaks of the theoretical isotope distribution
    and the peaks in their confidence intervals.

    Parameters
    ----------
    spectrum : Spectrum
        The spectrum, where algorithm tries to detect the substance.
    formula : Formula
        The Formula class of the substance.
    cal_error : float
        The radius of the first peak' vicinity (default is 0.006).
    dist_error : float
        The possible error in distance between the peaks, characterizes the radius of the second and the next
        peak's vicinity (default is 0.001).
    distance : int
        The parameter, used in peak finding algorithm. (default is 50).
    max_peaks : int
        Limits the number of possible first peaks (default is 5).
     
    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, float, float]
    Cosine distance, the arrays of possible masses and intensities and matched peaks percentage.
    Mass error (in ppm)
    """
    teor_masses, teor_peaks = formula.isodistribution()
    deisotoped_masses, deisotoped_peaks = del_isotopologues(teor_masses, teor_peaks)
    distances = np.diff(deisotoped_masses)

    first_peak_mass = deisotoped_masses[0]
    #import ipdb; ipdb.set_trace()
    pos_first_peak_masses, pos_first_peak_ints, pos_peak_indices, is_mtchd_f = get_peak_candidates(spectrum,
                                                                                                   first_peak_mass,
                                                                                                   cal_error,
                                                                                                   distance=distance)

    if len(pos_first_peak_masses) > max_peaks:
        pos_first_peak_masses = pos_first_peak_masses[np.argsort(pos_first_peak_ints)][::-1][0:max_peaks]
        pos_first_peak_ints = np.sort(pos_first_peak_ints)[::-1][0:max_peaks]
        pos_peak_indices = pos_peak_indices[np.argsort(pos_first_peak_ints)][::-1][0:max_peaks]

    pos_cosines = {}
    vector_indices = {}
    matched_percentages = {}
    for i in range(len(pos_peak_indices)):
        pos_first_peak_mass = pos_first_peak_masses[i]
        pos_first_peak_int = pos_first_peak_ints[i]
        vector = np.array([[pos_first_peak_mass], [pos_first_peak_int]])
        j = vector.shape[1] + 1
        mtchd_p_per = [is_mtchd_f]
        for delta in distances:
            teor_vector = deisotoped_peaks[0:j]
            j += 1
            vector, is_matched = add_new_peak(spectrum, delta, vector, teor_vector, dist_error, distance)
            mtchd_p_per.append(is_matched)

        if len(vector[1]) != len(deisotoped_peaks):
            raise ValueError(
                "The number of components of the possible intensities vector should be equal to the real deisotoped" +
                " masses' vector.")

        vector_indices[i] = vector
        pos_cosines[i] = cosine(vector[1], deisotoped_peaks)
        matched_percentages[i] = np.array(mtchd_p_per).mean()
        real_coords = vector_indices[min(pos_cosines, key=pos_cosines.get)]
        mass_delta = abs((real_coords[0] - deisotoped_masses)/deisotoped_masses).mean()*10**6
    return min(pos_cosines.values()), real_coords, \
           matched_percentages[min(pos_cosines, key=pos_cosines.get)], mass_delta
