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
name_decoder = df['Vector_name'].map(lambda x: take_code(x))
required_keys = df['Vector_name']
colormapper = {
    'white': 'red',
    'green': 'green',
    'oolong': 'blue',
    'black': 'yellow'
}

root = os.path.abspath('.')
spec_vecs_path = os.path.join(root, 'data', 'plot_pca_files', 'spec_vecs_dictionary.pkl')
pickle_file = open(spec_vecs_path, 'rb')
spec_vecs = pkl.load(pickle_file)
filenames = ['Tea_14_31_01_167.mzXML',
             'Tea_10_27_01_156.mzXML',
             'Tea_17_34_01_177.mzXML',
             'Tea_5_22_01_140.mzXML', 
             'Tea_6_23_01_143.mzXML',
             'Tea_9_26_01_152.mzXML',
             'Tea_11_28_01_159.mzXML',
             'Tea_2_19_01_131.mzXML',
             'Tea_4_21_01_137.mzXML']

agg_decoder = ['Green', 'Green', 'Green', 'Pu-erh', 'Black', 'Black', 'Black', 'Massala', 'Cham']


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
