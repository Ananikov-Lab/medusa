from argparse import ArgumentParser
from mass_automation.formula import Formula


def parse_formulas(path):
    formulas = []

    with open(path, 'r') as f:
        for line in f:
            try:
                formulas.append(Formula(line.strip().replace('+', '').replace('-', '')).dict_formula)
            except:
                continue

    return formulas


def formula2str(formula):
    return ''.join([k + str(int(v)) for k, v in formula.items()])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('formula_list', type=str)
    args = parser.parse_args()

    formulas = parse_formulas(args.formula_list)

    for formula in formulas:
        any_element = True
        for element in formula:
            if element not in ['C', 'H', 'N', 'O', 'P', 'S']:
                any_element = False

        if any_element:
            print(formula2str(formula))
