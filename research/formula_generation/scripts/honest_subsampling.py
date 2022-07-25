import argparse
from collections import defaultdict

import numpy as np

from mass_automation.formula import Formula


def parse_formulas(path):
    formulas = []
    elements = set()

    with open(path, 'r') as f:
        for line in f:
            try:
                formulas.append(Formula(line.strip().replace('+', '').replace('-', '')).dict_formula)
                elements.update(formulas[-1].keys())
            except:
                continue

    return formulas, elements


def formula2str(formula):
    return ''.join([k + str(int(v)) for k, v in formula.items()])


def compute_stats(formulas):
    total_dictionary = defaultdict(int)
    for formula in formulas:
        for k, v in formula.items():
            total_dictionary[k] += 1

    return total_dictionary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample list of formulas in a way in which rare elements are preserved")
    parser.add_argument('formula_list', type=str)
    parser.add_argument('element_threshold', type=float, default=10_000)
    parser.add_argument('others_percent', type=float, default=0.1)

    args = parser.parse_args()

    formulas, elements = parse_formulas(args.formula_list)

    element_statistics = compute_stats(formulas)
    elements2keep = [k for k, v in element_statistics.items() if v <= args.element_threshold]

    for formula in formulas:
        any_element = False
        for element in elements2keep:
            if element in formula:
                any_element = True

        if any_element:
            print(formula2str(formula))
        else:
            if np.random.random() < args.others_percent:
                print(formula2str(formula))
