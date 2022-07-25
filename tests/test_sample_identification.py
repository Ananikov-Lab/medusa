import os
import pickle as pkl

import pandas as pd

from mass_automation.sample_identification.plot import plot_pca
from mass_automation.sample_identification.plot import agglosamples
from .utils import not_raises


def take_code(filename):
    """Returns tea code by mzXML filename.
    """
    code = filename.split('_')[0]
    return code


root = os.path.abspath('.')
path = os.path.join(root, 'data', 'pca_tables', 'Tea_table.csv')
df = pd.read_csv(path, encoding='Windows-1251', sep=';')
class_decoder = df['Type']
name_decoder = df['mzXML'].map(lambda x: take_code(x))
required_keys = df['mzXML']
colormapper = {
    'white': 'red',
    'green': 'green',
    'oolong': 'blue',
    'black': 'yellow'
}

root = os.path.abspath('.')
spec_vecs_path = os.path.join(root, 'data', 'plot_pca_files', 'spectra_vecs_dictionary.pkl')
pickle_file = open(spec_vecs_path, 'rb')
spec_vecs = pkl.load(pickle_file)
filenames = ['UM-2_23_01_487.mzXML',
             'UM-2_23_01_488.mzXML',
             'P-5_16_01_467.mzXML',
             'P-5_16_01_466.mzXML',
             'KD-2_2_01_422.mzXML',
             'P-3_14_01_461.mzXML',
             'UM-6_27_01_500.mzXML',
             'UM-1_22_01_484.mzXML',
             'P-2_13_01_456.mzXML',
             'P-2_13_01_457.mzXML',
             'UM-5_26_01_495.mzXML',
             'UM-5_26_01_496.mzXML']

agg_decoder = ['White 1',
               'White 2',
               'Oolong 1',
               'Oolong 2',
               'Oolong 3',
               'Green 1',
               'Green 2',
               'Green 3',
               'Black 1',
               'Black 2',
               'Black 3',
               'Black 4']


class TestPlotPCA:
    def test_no_exceptions_pca(self):
        with not_raises(Exception):
            plot_pca(spec_vecs,
                     required_keys,
                     class_decoder,
                     name_decoder,
                     colormapper,
                     IsNameDecoder=True,
                     show=False)

    def test_no_exceptions_tsne(self):
        with not_raises(Exception):
            plot_pca(spec_vecs,
                     required_keys,
                     class_decoder,
                     name_decoder,
                     colormapper,
                     dim_red='TSNE',
                     IsNameDecoder=True,
                     show=False)

    def test_no_exceptions_low_bins(self):
        with not_raises(Exception):
            plot_pca(spec_vecs,
                     required_keys,
                     class_decoder,
                     name_decoder,
                     colormapper,
                     dim_red='TSNE',
                     max_bin=200,
                     IsNameDecoder=True,
                     show=False)


class TestAggloClustering:
    def test_agglosamples(self):
        with not_raises(Exception):
            agglosamples(spec_vecs,
                         filenames,
                         agg_decoder,
                         show=False)
