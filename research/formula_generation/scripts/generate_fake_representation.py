""" Generates fake representations for a list of formulas

The script takes list of formulas as an input (one formula — one line). The formulas must be correct, and do not
contain any brackets or so. The formula generation algorithm assumes positive charge.

See more details in ``mass_automation.formula.data.generate_fake_representation`` function.

"""
import gzip
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from pyteomics.mass import calculate_mass

from mass_automation.formula.data import generate_fake_representation
from mass_automation.formula import Formula

BIG_NUMBER = 1_0000_000

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate isotopic distributions")
    parser.add_argument('formula_list', type=str, help="File with list of formulas, each line represents new formula")
    parser.add_argument('output', type=str, help="Output file")
    args = parser.parse_args()

    with open(args.formula_list, 'r') as fin:
        with gzip.open(args.output, 'wt') as fout:
            for line in tqdm(fin):

                try:
                    formula = Formula(line.strip())

                    if calculate_mass(formula.str_formula) > 2000:
                        continue

                    representation = generate_fake_representation(formula, mode='middle',
                                                                  vectorization_method='convolution',
                                                                  sigma=np.random.uniform(5, 10) * 1e-4)
                    fout.write(
                        line.strip() + '\t' + '\t'.join(
                            [np.array_str(repr[0], max_line_width=BIG_NUMBER) + '\t' + str(repr[1]) for repr in
                             representation]) + '\n'
                    )
                except Exception as e:
                    print(e)