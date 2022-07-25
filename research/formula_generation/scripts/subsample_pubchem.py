import os
import argparse
import gzip

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem


def process_file(path, subsampling_rate, tag):
    errors = 0
    attempts = 0

    inf = gzip.open(path)
    gzsuppl = Chem.ForwardSDMolSupplier(inf)

    with gzip.open(f"{path}.{tag}.gz", 'wt') as fout:
        for molecule in gzsuppl:
            if np.random.random() > 1 - subsampling_rate:
                attempts += 1

                try:
                    fout.write(
                        molecule.GetPropsAsDict()['PUBCHEM_MOLECULAR_FORMULA'] + '\n'
                    )
                except Exception as e:
                    print(e)
                    errors += 1

    return path, errors, attempts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script subsamples PubChem to generate list of formulas")
    parser.add_argument('path', type=str, help="Path to the PubChem's SDF files")
    parser.add_argument('subsampling_rate', type=float, help="What part of the dataset to subsample")
    parser.add_argument('n_processes', type=int, help="Number of processes")
    parser.add_argument('tag', type=str, help="Tag for the output file")

    args = parser.parse_args()

    files = os.listdir(args.path)
    files = [file for file in files if file.endswith('sdf.gz')]

    results = Parallel(n_jobs=args.n_processes, verbose=10)(
        delayed(process_file)(
            os.path.join(args.path, filename), args.subsampling_rate, args.tag
        ) for filename in files
    )

    with open(f"report.{args.tag}.txt", 'w') as f:
        for path, errors, attempts in results:
            f.write(f"{path}\t{errors}\t{round(errors / attempts if attempts != 0 else 0, 3)}\n")

    os.system(
        f"zcat {args.path}/*.{args.tag}.gz > {args.tag}.merged.txt"
    )
