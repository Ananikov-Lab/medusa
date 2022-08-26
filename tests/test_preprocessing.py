import os
import gc
import pickle as pkl

import numpy as np

from mass_automation.experiment import Experiment, Spectrum
from mass_automation.plot import plot_spectrum
from .utils import not_raises


class TestExperiment:
    def test_no_exceptions(self):
        with not_raises(Exception):
            root = os.path.abspath('..')
            path = os.path.join(root, 'data', 'testing_spectra', 'test_experiment.mzXML')
            exp = Experiment(path, 32, 2)
            del exp
            gc.collect()

    def test_summarize(self):
        with not_raises(Exception):
            root = os.path.abspath('..')
            path = os.path.join(root, 'data', 'testing_spectra', 'test_summarize.mzXML')
            exp = Experiment(path, 16, 2)
            exp.summarize(0, exp.len)
            del exp
            gc.collect()

class TestSpectrum:
    def test_no_exceptions(self):
        with not_raises(Exception):
            masses = np.random.random((10000,))
            ints = np.random.random((10000,))
            Spectrum(masses, ints, 64, 16)

    def test_vectorize(self):
        root = os.path.abspath('..')
        path = os.path.join(root, 'data', 'testing_spectra', 'test_experiment.mzXML')
        exp = Experiment(path, 32, 2)
        spec = exp[0]
        output_vector = spec.vectorize()
        assert len(output_vector) == 850
        assert max(output_vector) == 1
        del exp, spec
        gc.collect()

    def test_plot_spectrum(self):
        root = os.path.abspath('..')
        path = os.path.join(root, 'data', 'testing_spectra', 'D3_3_01_270_C27H36N2H.pkl')
        with open(path, 'rb') as f:
            spectrum = pkl.load(f)
        with not_raises(Exception):
            plot_spectrum(spectrum, show=False)
