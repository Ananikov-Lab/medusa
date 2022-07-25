import os
import pickle as pkl
from typing import List

import numpy as np
import gc
from pytest import approx

from mass_automation.formula import Formula
from mass_automation.formula.check_formula import check_presence, del_isotopologues
from mass_automation.formula.plot import plot_compare
from .utils import not_raises


class TestFormula:
    def test_formula_one(self):
        assert Formula('CH4').dict_formula['C'] == 1

    def test_formula_four(self):
        assert Formula('CH4').dict_formula['H'] == 4

    def test_formula_two_instances(self):
        assert Formula('CH4C1AuO8C2').dict_formula['H'] == 4

    def test_formula_dict(self):
        assert Formula({'C': 1, 'H': 4}).str_formula in ['CH4', 'H4C']

    def test_isotopic_distribution(self):
        assert len(Formula('Pd').isodistribution()[0]) == 6

    def test_isotopic_distribution_two_elements(self):
        assert len(
            Formula('PdC').isodistribution(side_threshold=0)[0]
        ) == 6 * 2

    def test_isotopic_distribution_many_elements(self):
        assert len(
            Formula('Br5').isodistribution(side_threshold=0)[0]
        ) == 6

    def test_isotopic_distribution_many_elements_precision(self):
        formula = Formula('Br5').isodistribution(side_threshold=0)

        assert approx(formula[0][np.argmax(formula[1])], 1e-6) == 398.58704712

    def test_vector(self):
        vector = Formula('H2He').vector()

        true_vector = np.zeros(119)
        true_vector[0] = 2
        true_vector[1] = 1

        print(vector)
        print(true_vector)

        assert (vector == true_vector).all()


class TestDelIsotopologues:
    def test_del_isotopologues(self):
        str_formulas = [
            'C200H400',
            'C7H10NNiO2',
            'C27H36N2PdCH3CNCl',
            'C10H14FeO4',
            'C10H14MnO4',
            'BrUH',
            'CuCl2',
            'Au',
            'C9H8O4',
            'CH3CN',
            'Cl2O3',
            'Br150'
        ]
        for str_formula in str_formulas:
            masses, ints = Formula(str_formula).isodistribution()
            del_masses, del_ints = del_isotopologues(masses, ints)
            assert (masses[ints == max(ints)] in del_masses) == True
            assert ((np.round(del_masses, 0) == np.unique(np.round(masses, 0))).all()) == True
            if len(del_masses) > 1:
                assert ((np.diff(del_masses) >= 0.5).all()) == True


class TestCheckFormula:
    def test_check_simple(self):
        root = os.path.abspath('.')
        path = os.path.join(root, 'data', 'mass_spectra', 'D3_3_01_270_C27H36N2H.pkl')
        with open(path, 'rb') as f:
            spec = pkl.load(f)
        formula = Formula('C27H36N2H')
        assert abs(check_presence(spec, formula)[0] - 0.0003) < 10 ** (-5)
        with not_raises(Exception):
            plot_compare(spec, formula, show=False)
        del spec
        gc.collect()

    def test_check_pd(self):
        root = os.path.abspath('.')
        path = os.path.join(root, 'data', 'mass_spectra', 'D1_1_01_268_C27H36N2PdC5H5NCH3CNCl.pkl')

        with open(path, 'rb') as f:
            spec = pkl.load(f)
        formula = Formula('C27H36N2PdC5H5NCH3CNCl')
        assert abs(check_presence(spec, formula)[0] - 0.0011) < 10 ** (-4)
        with not_raises(Exception):
            plot_compare(spec, formula, show=False)
        del spec
        gc.collect()

    def test_check_ni(self):
        root = os.path.abspath('.')
        path = os.path.join(root, 'data', 'mass_spectra', 'D18_18_01_206_C7H10NNiO2.pkl')
        with open(path, 'rb') as f:
            spec = pkl.load(f)
        formula = Formula('C7H10NNiO2')
        assert abs(check_presence(spec, formula)[0] - 0.0009) < 10 ** (-5)
        with not_raises(Exception):
            plot_compare(spec, formula, show=False)
        del spec
        gc.collect()

    def test_check_mn(self):
        root = os.path.abspath('.')
        path = os.path.join(root, 'data', 'mass_spectra', 'D24_24_01_212_C25H35Mn2O10.pkl')
        with open(path, 'rb') as f:
            spec = pkl.load(f)
        formula = Formula('C25H35Mn2O10')
        assert abs(check_presence(spec, formula)[0] - 0.00071) < 10 ** (-5)
        with not_raises(Exception):
            plot_compare(spec, formula, show=False)
        del spec
        gc.collect()
