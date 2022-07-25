from typing import Union, Tuple, Optional, List, Callable

import numpy as np
import chemparse
from pyteomics.mass.mass import isotopologues, isotopic_composition_abundance, calculate_mass

from ..experiment import Spectrum
from ..utils import Element

ELECTRON_MASS = 0.00054858
NEUTRON_MASS = 1.00866492


class Formula:
    """A class used to represent Formula.

    Attributes
    ----------
    str_formula : str
        The molecular formula of the substance. Example: 'C2H5OH'.
    dict_formula : dict
        The dictionary, which corresponds to the formula.
        Example::
            {
                'C' : 2,
                'H' : 6,
                'O' : 1
            }
    formula : str or dict
        Dictionary or string, depending on the argument in __init__.
    charge : int
        The charge of the molecule.
    monoisotopic_mass : float
        The mass of the monoisotopic peak.
    """

    def __init__(self, formula: Union[str, dict], charge=None):
        """
        Parameters
        ----------
        formula : str or dict
            String or Dictionary. An Encoder of the molecule.
        charge : int
            The charge of the molecule (default is 1)
        """
        if type(formula) == str:
            self.str_formula = formula
            dict_formula = chemparse.parse_formula(formula)
            dict_formula2 = {k: int(v) for k, v in dict_formula.items()}
            self.dict_formula = dict_formula2

        elif type(formula) == dict:
            self.dict_formula = formula
            formula_list = []
            for key in formula.keys():
                formula_list.append(key)
                if formula[key] != 1:
                    formula_list.append(str(formula[key]))
            self.str_formula = ''.join(formula_list)
        else:
            raise TypeError('Parameter formula should be "str" or "dict"')
        self.formula = formula
        self.charge = charge if charge is not None else 1
        self.monoisotopic_mass = (calculate_mass(formula) - self.charge * ELECTRON_MASS) / abs(self.charge)

    def __add__(self, other):
        dict_add_formula = self.dict_formula
        for key in other.dict_formula.keys():
            if key not in self.dict_formula.keys():
                dict_add_formula[key] = other.dict_formula[key]
            else:
                dict_add_formula[key] += other.dict_formula[key]

        return Formula(dict_add_formula)

    def __sub__(self, other):
        dict_sub_formula = self.dict_formula
        for key in other.dict_formula.keys():
            if key not in self.dict_formula.keys():
                dict_sub_formula[key] = -other.dict_formula[key]
            else:
                dict_sub_formula[key] -= other.dict_formula[key]

        return Formula(dict_sub_formula)

    def vector(self):
        """Converts formula into a vector of quantities with length of number elements in the periodic table

        **Attention:** may not work for elements like Fl, as pyteomics does not recognize them.

        Returns
        -------
        np.ndarray
            Resulting vector
        """
        out = np.zeros(Element.n_elements)

        for element, n in self.dict_formula.items():
            out[getattr(Element, element) - 1] = n

        return out

    def isodistribution(self, side_threshold=0.001) -> Tuple[np.ndarray, np.ndarray]:
        """ Creates isotopic distribution.

        Gives back a tuple with masses and peaks of the theoretical isotopic distribution.

        Parameters
        ----------
        side_threshold: float
            Minimal required relative intensity of side isotopologues to the most intensive. (default is 0.001)

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]

        Raises
        ------
        ValueError
            Raises when isotopic distribution can not be constructed because of specific formula names.
        """
        isotopes = [[], []]
        for isotope in isotopologues(formula=self.str_formula):
            isotopes[0].append(calculate_mass(isotope) - self.charge * ELECTRON_MASS)
            isotopes[1].append(isotopic_composition_abundance(isotope))

        if len(isotopes[0]) == 0:
            raise ValueError("Isotopic distribution can't be constructed because of specific formula name.")

        isotopes[0] = np.array(isotopes[0]) / abs(self.charge)
        isotopes[1] = np.array(isotopes[1])

        isotopes[1] = isotopes[1][np.argsort(isotopes[0])]
        isotopes[0] = np.sort(isotopes[0])

        greater_than_thresh = np.where(isotopes[1] > side_threshold)[0]

        pos_first = greater_than_thresh[0]
        pos_last = greater_than_thresh[-1]

        isotopes[1] = isotopes[1][pos_first:pos_last + 1]
        isotopes[0] = isotopes[0][pos_first:pos_last + 1]

        teor_masses = isotopes[0]
        teor_peaks = isotopes[1] / isotopes[1].max()

        return teor_masses, teor_peaks


class RealIsotopicDistribution:
    def __init__(self, spectrum: Spectrum, peak_indices: List[int]):
        self.spectrum = spectrum
        self.peak_indices = sorted(peak_indices)
        self.monoisotopic = self.get_monoisotopic()

    def get_monoisotopic(self) -> int:
        """Find monoisotopic peak

        Currently "monoisotopic" is the first one. **THAT IS NO TRUE FOR SOME ELEMENTS.**
        Rewrite ``get_representation`` method accordingly if changing

        Returns
        -------
        int
            Index to the monoisotopic peak
        """
        return 0

    def get_representation(
            self, delta: Optional[float] = 0.025, length: Optional[int] = 100, f: Optional[Callable] = np.max,
            mode='middle', vectorization_method='simple', sigma: Optional[float] = None) -> List[
        Tuple[np.ndarray, float]]:
        """Calculates vector representation for the isotopic distribution

        Parameters
        ----------
        delta : float
            Size of peak vicinity
        length : int
            Number of items in resulting feature vector¬
        mode : str
            Mode of vectorization. One of the following:

            * ``middle`` — the most intensive in the middle
            * ``monoisotopic`` — some part of the spectrum left to the monoisotopic peak, incremented, by neuron mass
        vectorization_method : str
            Method of vectorization. One of the folloring:

            * ``simple``
            * ``convolution``
        sigma : float
            A parameter for the vectorization method

        Returns
        -------
        List[Tuple[np.ndarray, float]]
            Representations for each peak
        """

        max_intensity = max([self.spectrum.ints[peak] for peak in self.peak_indices])

        if mode == 'middle':
            out = []

            for peak in self.peak_indices:
                peak_mass = self.spectrum.masses[peak]

                sub_spectrum = self.spectrum.get_slice(
                    peak_mass - delta / 2, peak_mass + delta / 2
                )

                if vectorization_method == 'simple':
                    sub_spectrum = sub_spectrum.vectorize(peak_mass - delta / 2, peak_mass + delta / 2,
                                                          None, f, False, length, max_intensity)
                elif vectorization_method == 'convolution':
                    sub_spectrum = sub_spectrum.vectorize_by_convolution(peak_mass - delta / 2, peak_mass + delta / 2,
                                                                         length, sigma, max_intensity)

                out.append((sub_spectrum, peak_mass))

            return out

        if mode == 'monoisotopic':
            # IMPORTANT
            # Assumes the first peak to be monoisotopic, although that is not true for many elements
            # (Fe, Pd, for example)

            out = []

            for i in range(len(self.peak_indices)):
                right_mass = self.spectrum.masses[self.peak_indices[self.monoisotopic]] - NEUTRON_MASS * (
                        self.monoisotopic - i)
                left_mass = right_mass - delta

                sub_spectrum = self.spectrum.get_slice(
                    left_mass, right_mass
                )
                if vectorization_method == 'simple':
                    sub_spectrum = sub_spectrum.vectorize(left_mass, right_mass, None, f, False, length, max_intensity)
                elif vectorization_method == 'convolution':
                    sub_spectrum = sub_spectrum.vectorize_by_convolution(left_mass, right_mass, length, sigma,
                                                                         max_intensity)

                out.append((sub_spectrum, right_mass))

            return out
