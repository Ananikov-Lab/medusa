from typing import Callable, List, Union

import numpy as np
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence

from mass_automation.formula import Formula, RealIsotopicDistribution
from mass_automation.formula.check_formula import del_isotopologues
from mass_automation.experiment import Spectrum

normalizers = {
    'max': np.max,
    'min': np.min,
    'mean': np.mean,
    'sum': np.sum,
    'one': lambda x: 1,
    'means': lambda x: np.clip(np.array(
        [3.60372658e+01, 2.88056417e-05, 9.15330276e-03, 3.08382820e-04,
         6.18755445e-02, 3.00645962e+01, 3.51274824e+00, 5.16170597e+00,
         1.31945515e+00, 1.37611278e-05, 3.95996347e-02, 3.43262684e-03,
         5.76452632e-03, 1.50951594e-01, 1.68682814e-01, 6.85634851e-01,
         4.73159939e-01, 1.12804177e-04, 1.19415522e-02, 2.99758953e-03,
         2.41936243e-04, 4.67379903e-03, 1.62352400e-03, 2.47604656e-03,
         2.08865036e-03, 6.78629940e-03, 3.97799024e-03, 4.54592053e-03,
         7.37428898e-03, 6.03908859e-03, 1.48310850e-03, 4.28765640e-03,
         4.68633603e-03, 1.26135405e-02, 1.61566749e-01, 1.87759651e-05,
         1.38155802e-03, 6.59421494e-04, 1.96829415e-03, 3.79745616e-03,
         3.73575720e-04, 1.59218127e-03, 9.99999975e-06, 4.42681834e-03,
         1.35899126e-03, 3.66707053e-03, 2.12250045e-03, 8.80074338e-04,
         1.20478508e-03, 1.06351869e-02, 2.25790101e-03, 2.57885060e-03,
         7.80835003e-02, 5.51335397e-05, 9.01387422e-04, 1.13081618e-03,
         4.58827970e-04, 4.71365056e-04, 2.39428831e-04, 4.38768620e-04,
         9.99999975e-06, 3.48501519e-04, 3.73575720e-04, 5.75422950e-04,
         2.39428831e-04, 3.23427346e-04, 1.84265606e-04, 2.31906568e-04,
         1.10296751e-04, 4.07425861e-04, 2.00563838e-04, 7.17092131e-04,
         2.89577176e-04, 2.00339803e-03, 4.09933302e-04, 6.59421494e-04,
         1.49940676e-03, 4.61612828e-03, 1.84417679e-03, 2.18643970e-03,
         8.39955639e-04, 1.55206257e-03, 1.34143932e-03, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 2.24384319e-04, 2.25370932e-05, 7.68494210e-04,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06, 9.99999975e-06,
         9.99999975e-06, 9.99999975e-06, 9.99999975e-06]
    ), 1e-2, 1)
}


def formula2vector(formula: Formula, normalizer: str):
    vectorized_formula = formula.vector()
    return vectorized_formula / normalizers[normalizer](vectorized_formula)


def formula2element_amount(formula: Formula, element_id: int) -> int:
    return formula.vector()[element_id]


def formula2element(formula: Formula, element_id: int) -> bool:
    return formula2element_amount(formula, element_id) > 0


converter_mapper = {
    'vector': formula2vector,
    'element_amount': formula2element_amount,
    'element': formula2element
}


def generate_fake_representation(formula: Formula, **kwargs):
    """Simulates isotopic distribution and constructs pseudo-spectrum
    """
    masses, ints = formula.isodistribution()
    fake_spectrum = Spectrum(
        masses, ints
    )

    peak_indices = del_isotopologues(masses, ints, return_ids=True)

    id = RealIsotopicDistribution(fake_spectrum, peak_indices=peak_indices)
    return id.get_representation(**kwargs)


class PadSequence:
    """Prepares a sequence as an input for LSTM network
    """

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        seqs, labels = zip(*batch)
        seqs = [torch.FloatTensor(seq) for seq in seqs]

        return pack_sequence(seqs), torch.FloatTensor(labels)


class PadSequenceConstant:
    """Prepares a sequence as an input for MLP
    """

    def __init__(self, first_x):
        self.first_x = first_x

    def __call__(self, batch):
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        seqs, labels = zip(*batch)

        padded_seqs = []

        for seq in seqs:
            zeros = torch.zeros(max(0, (self.first_x - len(seq))) * len(seq[0]), dtype=torch.float32)
            reduced = [torch.FloatTensor(subseq) for subseq in seq[:self.first_x]]

            padded_seqs.append(torch.cat(reduced + [zeros], dim=0))

        return torch.vstack(padded_seqs), torch.FloatTensor(labels)


class OriginalFormulaDataset(Dataset):
    """Constructs the dataset of model training
    """

    def __init__(self, formulas: List[Formula], formula_converter: Callable,
                 representations: List[str], augmentations: Union[Callable, None], is_classifier: bool, **kwargs):
        self.formulas = [formula_converter(formula, **kwargs) for formula in formulas]

        self.representations = []
        for representation in tqdm(representations):
            representation = representation.split('\t')

            self.representations.append([])
            for i in range(0, len(representation), 2):
                self.representations[-1].append(
                    np.append(np.fromstring(representation[i][1:-1], dtype=np.float, sep=' ')[:-1],
                              float(representation[i + 1]) / 1000))

        self.augmentations = augmentations
        self.is_classifier = is_classifier

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, item):
        augmented_repr = [self.augmentations(repr) if self.augmentations is not None else repr for repr in
                          self.representations[item]]
        augmented_repr = [repr for repr in augmented_repr if repr is not None]  # Needed to support the DroppingAug

        return augmented_repr, \
               (self.formulas[item] > 0).astype(float) if self.is_classifier else self.formulas[item].astype(float)
