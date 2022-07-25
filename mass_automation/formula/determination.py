from typing import List

import numpy as np
from tqdm.autonotebook import trange

from pyteomics.mass import calculate_mass

from . import Formula
from .check_formula import check_presence
from ..experiment import Spectrum


def brute_force_search(exact_mass: float, elements: List[str], low_limits, high_limits, spectrum: Spectrum,
                       evaluation_threshold: float,
                       check_presence_params: dict, return_threshold: float):
    def get_subformula(elements, low_limits, high_limits, current_formula, current_mass, target_mass, threshold):

        element_masses = [calculate_mass(element + '1') for element in elements]  # Really terrible way to do that

        max_count = int((target_mass + threshold - current_mass) // element_masses[0] + 1)
        if len(elements) > 1:
            range_ = range(low_limits[0], min(max_count, high_limits[0]))
        else:
            range_ = range(low_limits[0], min(max_count, high_limits[0]))

        for i in range_:
            mass = element_masses[0] * i + current_mass

            current_formula_ = current_formula.copy()
            current_formula_[-len(elements)] = i

            if mass > target_mass:
                break

            if len(element_masses) != 1:
                yield from get_subformula(elements[1:], low_limits[1:], high_limits[1:], current_formula_, mass,
                                           target_mass, threshold)
            else:
                if (mass <= target_mass + threshold) & (mass >= target_mass - threshold):
                    yield [mass, current_formula_]

    for mass, formula in get_subformula(elements, low_limits, high_limits, np.zeros(len(elements)), 0, exact_mass,
                                         evaluation_threshold):
        cosine = check_presence(
            spectrum,
            Formula({el: int(q) for el, q in zip(elements, formula)}),
            **check_presence_params
        )[0]

        if cosine < return_threshold:
            yield [formula, mass, cosine, 1e6 * (exact_mass - mass) / mass]
