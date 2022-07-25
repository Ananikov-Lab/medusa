from __future__ import annotations

import os
import pickle as pkl
from pathlib import Path
from typing import Optional, Callable, Hashable

import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzXMLFile
from tqdm.autonotebook import tqdm

from .utils import lorentzian


def peak_pick(mzs, hs, min_distance=0.01, algorithm='std', alpha=None, verbose=False, threshold=None):
    if threshold is None:
        if algorithm is None:
            raise ValueError('Set algorithm or minimal intensity threshold value')
        elif algorithm == 'std':
            threshold = np.median(hs) + 2 * np.std(hs)
        elif algorithm == 'hist':
            hist, bin_edges = np.histogram(hs, bins=int(len(hs) / 100))
            threshold = bin_edges[1]
        elif algorithm == 'quantile':
            if alpha:
                threshold = np.quantile(hs, alpha)
            else:
                threshold = np.quantile(hs, 0.995)
        else:
            raise ValueError('Unknown algorithm')

        cand_peaks = [[], [], []]
        zero_out = np.where(hs > threshold, hs, 0)
        cand_peaks[0] = mzs[zero_out != 0]
        cand_peaks[1] = hs[zero_out != 0]
        cand_peaks[2] = np.arange(len(mzs))[zero_out != 0]

        peaks_to_be_selected = [[], [], []]

        for mass, int_, i in tqdm(zip(*cand_peaks), total=len(cand_peaks[0]), disable=(not verbose)):
            mass_low = cand_peaks[0] > mass - min_distance
            mass_high = cand_peaks[0] < mass + min_distance

            if cand_peaks[1][mass_low & mass_high].max() == int_:
                if (i != 0) and (i != (len(hs) - 1)):
                    if (hs[i - 1] < hs[i]) and (hs[i] > hs[i + 1]):
                        peaks_to_be_selected[0].append(mass)
                        peaks_to_be_selected[1].append(int_)
                        peaks_to_be_selected[2].append(i)

        peaks_to_be_selected = [np.array(peaks_to_be_selected[0]),
                                np.array(peaks_to_be_selected[1]),
                                np.array(peaks_to_be_selected[2])]

        return peaks_to_be_selected


class Spectrum:
    """A class used to represent Spectrum

    Attributes
    ----------
    masses : np.ndarray
        The array of masses.
    ints : np.ndarray
        The array of intensities.
    n_scans : int
        The number of scans.
    n_points : int
        The number of millions of points.
    deisotoped : bool
        If True, the spectrum has been deisotoped.
    isotopic_distributions : np.ndarray
        The classifier labels of the spectrum.
    """

    def __init__(self, masses: np.ndarray, ints: np.ndarray,
                 n_scans: Optional[int] = None, n_points: Optional[int] = None, path: Optional[str] = None):
        """
        Parameters
        ----------
        masses : np.ndarray
            The array of masses of the spectrum
        ints : np.ndarray
            The array of intensities of the spectrum
        n_scans : int
            The number of scans
        n_points : int
            The number of points
        path : str
            The path to the caching file
        """
        self.masses = masses
        self.ints = ints
        self.n_scans = n_scans
        self.n_points = n_points

        self.deisotoped = False
        self.isotopic_distributions = None

        self.path = path

    def vectorize(self, min_mass: Optional[int] = 150, max_mass: Optional[int] = 1000, delta_mass: Optional[int] = 1,
                  method: Optional[Callable] = np.max, keep_state: Optional[bool] = True,
                  n_bins: Optional[int] = None, normalize: Optional[float] = None) -> np.ndarray:
        """
        Performs vectorization of the spectrum, where vector components are encoded by
        one of the available methods.

        Parameters
        ----------
        min_mass : int
            The left margin of the interval where the vectorization is performed (default is 150).
        max_mass : int
            The right margin of the interval where the vectorization is performed (default is 1000).
        delta_mass : int
            The length of the interval which is characterized by one spectrum's vector component (default is 1).
        method : Callable
            The method of vectorization. For instance, the following NumPy functions may be used

            - np.max - by maximal intensity in vector component intervals,
            - np.sum - by total intensity
            - np.mean - by mean intensity (default is np.max).
        keep_state : bool
            Defines whether results will be cached or not.
        n_bins : int
            Number of bins. If ``None`` the number of bins will be calculated from ``delta_mass`` parameter.
        normalize: float
            If ``None``, maximum value is used for normalization. If ``-1``, the spectrum is not normalized.
            Other numbers are used as a normalization constants.

        Returns
        -------
        np.ndarray
            The vector, where components are the numbers characterizing the spectrum intervals.
        """
        if not n_bins:
            n_bins = int((max_mass - min_mass) / delta_mass + 1)

        masses = self.masses
        rel_ints = self.ints / (self.ints.max() if not normalize else (1 if normalize == -1 else normalize))

        indices = np.where(masses < max_mass)[0]
        masses = masses[indices]
        rel_ints = rel_ints[indices]
        x = np.linspace(min_mass, max_mass, n_bins)
        output_vector = np.zeros(n_bins - 1)

        for i in range(n_bins - 1):
            slice = (masses >= x[i]) & (masses < x[i + 1])
            output_vector[i] = method(rel_ints[slice]) if slice.any() else 0

        if keep_state:
            self.save_state()

        self.vectorized = output_vector

        return output_vector

    def vectorize_by_convolution(self, min_mass: float, max_mass: float, n_bins: int, sigma: 1e-5,
                                 normalize: Optional[float] = None) -> np.ndarray:
        """
        Performs vectorization via convolution. Each peak is represented as a gaussian curve

        Parameters
        ----------
        min_mass : float
            The left margin of the interval where the vectorization is performed.
        max_mass : float
            The right margin of the interval where the vectorization is performed.
        n_bins : int
            The number of bins.
        sigma : float
            The width of the lorentzian curve.
        normalize : float
            If ``None``, maximum value is used for normalization. If ``-1``, the spectrum is not normalized. Other
            numbers are used as a normalization constants.

        Returns
        -------
        np.ndarray

        """
        x = np.linspace(min_mass, max_mass, n_bins)
        out = np.zeros(n_bins)

        max_int = self.ints[(self.masses >= min_mass) & (self.masses <= max_mass)].max()
        normalizing_constant = max_int if not normalize else (1 if normalize == -1 else normalize)

        # TODO: can be optimized
        for mass, int in zip(self.masses, self.ints):
            out += (int / normalizing_constant) * lorentzian(x, mass, sigma)

        return out

    def save_state(self):
        """Saves current state of the object into caching pickle file, set in ```self.path```
        """
        if self.path:
            with open(self.path, 'wb') as f:
                pkl.dump(self, f)

    def get_slice(self, left_mass: float, right_mass: float) -> Spectrum:
        """
        Returns a subspectrom of the original spectrum within a specific mass interval ```[left_mass, right_mass]```

        Parameters
        ----------
        left_mass : float
            The left limit of the interval
        right_mass : float
            The right limit of the interval

        Returns
        -------
        Spectrum
            Subspectrum of the original spectrum. No caching applied
        """
        left_index = np.where(self.masses > left_mass)[0]
        left_index = left_index[0] if len(left_index) != 0 else 0

        right_index = np.where(self.masses > right_mass)[0]
        right_index = right_index[0] if len(right_index) != 0 else len(self.masses)

        sub_spectrum = Spectrum(
            self.masses[left_index: right_index], self.ints[left_index: right_index]
        )

        return sub_spectrum

    def to_msi_warp(self, min_distance=0.01, algorithm="std", alpha=None):
        mzs, hs, _ = peak_pick(self.masses, self.ints, min_distance=min_distance, algorithm=algorithm, alpha=alpha)
        return mzs, hs


class Experiment:
    """A class used to represent Experiment.

    Attributes
    ----------
    spectra_mass : List
        The list of spectra masses.
    spectra_ints : List
        The list of spectra intensities
    path : str
        The path to the file.
    format : str
        The format of the file (default is 'mzXML').
    n_scans : int
        The number of scans in the experiment.
    n_points : int
        The number of points in the experiment (in millions).
    """

    def __init__(self, path: str, n_scans=None, n_points=None, format: Optional[str] = 'mzXML',
                 verbose: Optional[bool] = True, suppress_caching=True):
        """
        Parameters
        ----------
        path : str
            The path to the file.
        n_scans : int
            The number of scans.
        n_points : int
            The number of points.
        format : str
            The format of the file (default is 'mzXML').
        verbose : bool
            If True, show TQDM bars.
        """
        self.spectra_mass = []
        self.spectra_ints = []
        self.path = path
        self.format = format
        self.n_scans = n_scans
        self.n_points = n_points
        self.suppress_caching = suppress_caching

        if format == 'mzXML':
            exp = MSExperiment()
            MzXMLFile().load(path, exp)

            for spectrum in tqdm(exp.getSpectra(), disable=(not verbose)):
                peaks_val, peaks_int = spectrum.get_peaks()
                self.spectra_mass.append(peaks_val)
                self.spectra_ints.append(peaks_int)

        elif format == 'mzML':
            raise NotImplementedError

        else:
            raise ValueError

        self.len = len(self.spectra_mass)

    def __getitem__(self, item: int) -> Spectrum:
        name = None
        if not self.suppress_caching:
            found_in_index = self.check_in_index('get_item', item)

            if found_in_index:
                with open(found_in_index, 'rb') as f:
                    return pkl.load(f)

            name = self.find_name()
            self.add_to_index('get_item', item, name)

        spec = Spectrum(
            self.spectra_mass[item],
            self.spectra_ints[item],
            self.n_scans,
            self.n_points,
            name
        )

        if not self.suppress_caching:
            spec.save_state()
        return spec

    def get_names(self):
        """Helper function to get parent dir for the experiment file and it's filename without extension
        """
        return Path(self.path).parent.absolute(), os.path.split(self.path)[1].rsplit('.', 1)[0]

    def find_name(self):
        """Looks for available name
        """

        parent, exp_name = self.get_names()

        files = [
            int(file.rsplit('.', 2)[1]) for file in os.listdir(parent) if file.startswith(
                exp_name + '.cache.'
            )
        ]

        if files:
            return os.path.join(parent, exp_name + '.cache.' + str(max(files) + 1) + '.pkl')
        else:
            return os.path.join(parent, exp_name + '.cache.0.pkl')

    def add_to_index(self, function: str, params: Hashable, name: str):
        """Adds item to the index

        Parameters
        ----------
        function : str
            name of the function, currently it is either "get_item" or "summarize"
        params : Hashable
            parameters, used for calling the function. **Must be immutable**
        name : str
            path to the cache file
        """
        parent, exp_name = self.get_names()
        index_path = os.path.join(parent, exp_name + '.index')

        if not os.path.exists(index_path):
            data = {}

        else:
            with open(index_path, 'rb') as f:
                data = pkl.load(f)

        data[(function, params)] = name

        with open(index_path, 'wb') as f:
            pkl.dump(data, f)

    def check_in_index(self, function: str, params: Hashable) -> str:
        """Checks presence of particular function in index

        Parameters
        ----------
        function : str
            name of the function, currently it is either "get_item" or "summarize"
        params : Hashable
            parameters, used for calling the function. **Must be immutable**

        Returns
        -------
        str
            path to the cache
        """
        parent, exp_name = self.get_names()
        index_path = os.path.join(parent, exp_name + '.index')

        if not os.path.exists(index_path):
            return None
        else:
            with open(index_path, 'rb') as f:
                data = pkl.load(f)

            if (function, params) in data:
                return data[(function, params)]

    def get_chromatogram(self):
        """Returns the array of total intensities of the spectra.

        Gives an opportunity to estimate the approximate amount of the sample in specific spectrum.

        Returns
        -------
        np.ndarray
            The array of total intensities of the spectra.
        """
        return np.arange(self.len), np.array([sum(ints) for ints in self.spectra_ints])

    def summarize(self, item1=None, item2=None, subtract: Optional[bool] = True, cache=False) -> Spectrum:
        """Summarizes the spectra of the experiment from a given interval.

        The right threshold is not included in the summation.

        Parameters
        ----------
        item1 : int
            The number of the first spectrum in summation.
        item2 : int
            The number of the last spectrum in summation + 1.
        subtract : bool
            If True, the mean intensities of the spectra, which go after the summation interval,
            are substracted from the sum.
        cache : bool
            If True, resulting spectrum is saved, else not.

        Returns
        -------
        Spectrum
            The resulted summarized spectrum.

        Raises
        ------
        ValueError
            If item1 or item2 exceeds the number of specta in the experiment or item1 is bigger than item2.
        """

        if item1 is None:
            item1 = 0
        if item2 is None:
            item2 = self.len

        if (item1 >= self.len) or (item2 > self.len) or (item1 > item2):
            raise ValueError

        name = None
        if cache:
            found_in_index = self.check_in_index('vectorize', (item1, item2))

            if found_in_index:
                with open(found_in_index, 'rb') as f:
                    return pkl.load(f)

            name = self.find_name()
            self.add_to_index('vectorize', (item1, item2), name)

        spectrum = Spectrum(self.spectra_mass[item1], np.zeros(len(self.spectra_ints[item1])), self.n_scans,
                            self.n_points, name)

        for item in range(item1, item2):
            ints = self.__getitem__(item).ints
            spectrum.ints = spectrum.ints + ints

        if subtract and (item2 != self.len):
            subtrspec = Spectrum(self.spectra_mass[item2], np.zeros(len(self.spectra_ints[item2])), self.n_scans,
                                 self.n_points, name)
            for item in range(item2, self.len):
                ints = self[item].ints
                subtrspec.ints = subtrspec.ints + ints
            subtrspec.ints = subtrspec.ints / (self.len - item2)
            spectrum.ints = spectrum.ints - subtrspec.ints

        if cache:
            spectrum.save_state()

        return spectrum

    def to_sima(self, path, min_distance=0.01, algorithm="std", alpha=None):
        total_mzs = []
        total_hs = []
        total_rts = []
        for i in range(self.len):
            spec = self.__getitem__(i)
            mzs, hs, _ = peak_pick(spec.masses, spec.ints,
                                   min_distance=min_distance,
                                   algorithm=algorithm,
                                   alpha=alpha)
            total_mzs.extend(list(mzs))
            total_hs.extend(list(hs))
            total_rts.extend(list(np.zeros(mzs.shape[0]) + i))

        total_charges = [1] * len(total_mzs)
        a = np.array(total_mzs)
        b = np.array(total_charges)
        c = np.array(total_hs)
        d = np.array(total_rts)
        df = pd.DataFrame(np.array([np.sort(a), b[np.argsort(a)], c[np.argsort(a)], d[np.argsort(a)]]).T)
        df.to_csv(path, index=False, header=False, sep=' ')

    def to_chrom_align_net(self):
        pass
