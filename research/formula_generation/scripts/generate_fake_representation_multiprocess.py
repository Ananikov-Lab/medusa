""" Generates fake representations for a list of formulas

The script takes list of formulas as an input (one formula — one line). The formulas must be correct, and do not
contain any brackets or so. The formula generation algorithm assumes positive charge.

See more details in ``mass_automation.formula.data.generate_fake_representation`` function.

"""
import gzip
import multiprocessing as mp
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from pyteomics.mass import calculate_mass

from mass_automation.formula.data import generate_fake_representation
from mass_automation.formula import Formula

BIG_NUMBER = 1_0000_000

def compute_representation_string(formula):
    try:
        representation = generate_fake_representation(formula, mode='middle',
                                                      vectorization_method='convolution', length=101,
                                                      sigma=np.random.uniform(.5, 3) * 1e-4)
        return formula.str_formula + '\t' + '\t'.join(
            [np.array_str(repr[0][1:], max_line_width=BIG_NUMBER) + '\t' + str(repr[1]) for repr in
             representation])
    except:
        return None


def worker(formula, q):
    res = compute_representation_string(formula)
    q.put(res)
    return res


def listener(q):
    '''listens for messages on the q, writes to file. '''

    with gzip.open('output.2021-08-08.gz', 'wt') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed\n')
                break

            if m is not None:
                f.write(str(m) + '\n')
                f.flush()


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate isotopic distribution in a multiprocess fashion")
    parser.add_argument('formula_list', type=str, help="File with list of formulas, each line represents new formula")
    args = parser.parse_args()

    formulas = []

    with open(args.formula_list, 'r') as fin:
        for line in tqdm(fin):

            try:
                formula = Formula(line.strip())
            except:
                continue

            if calculate_mass(formula=formula.str_formula) > 2000:
                continue

            formulas.append(formula)

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(listener, (q,))

    jobs = []
    for formula in formulas:
        job = pool.apply_async(worker, (formula, q))
        jobs.append(job)

    for job in jobs:
        job.get()

    q.put('kill')
    pool.join()
    pool.close()
