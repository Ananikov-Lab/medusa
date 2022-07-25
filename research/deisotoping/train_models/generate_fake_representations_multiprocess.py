""" Generates fake representations for a list of formulas

The script takes list of formulas as an input (one formula — one line). The formulas must be correct, and do not
contain any brackets or so. The formula generation algorithm assumes positive charge.

See more details in ``mass_automation.formula.data.generate_fake_representation`` function.

"""
import gzip
import multiprocessing as mp
from argparse import ArgumentParser
from random import shuffle

import numpy as np
from tqdm import tqdm
from pyteomics.mass import calculate_mass

from mass_automation.formula import Formula
from mass_automation.formula.check_formula import del_isotopologues


def compute_representation_string(formula):
    try:
        masses, ints = formula.isodistribution()

        if len(masses) == 0:
            print(formula.str_formula)

        masses, ints = del_isotopologues(masses, ints)
        return formula.str_formula+'\t'+'\t'.join([str(mass) for mass in masses])
    except:
        return None


def worker(formula, q):
    res = compute_representation_string(formula)
    q.put(res)
    return res


def listener(q):
    '''listens for messages on the q, writes to file. '''

    with gzip.open('output.gz', 'wt') as f:
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed\n')
                break

            if m is not None:
                f.write(str(m) + '\n')
                f.flush()


if __name__ == "__main__":
    parser = ArgumentParser()
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
    pool = mp.Pool(mp.cpu_count() + 20)

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
