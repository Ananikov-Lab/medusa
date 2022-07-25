import collections
from copy import copy
from typing import List, Optional

import numpy as np


class Reassigner:
    def __init__(self, rules: List[List[int]], method: str, replace_value: Optional[float] = int):
        self.rules = rules
        self.method = method
        self.replace_value = replace_value

    def __call__(self, vector: np.ndarray):
        out = copy(vector)

        for source, destination in self.rules:
            if self.method == 'replace':
                out[destination] = out[source]
            elif self.method == 'sum':
                out[destination] += out[source]
            else:
                raise ValueError('This transformation method is unknown')

            out[source] = self.replace_value

        return out


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


monoiotopic_conversion_rules = [[item - 1 for item in rule] for rule in [
    [37, 11],
    [55, 11],

    [39, 21],
    [41, 23],
    [43, 25],
    [75, 24],
    [45, 27],

    [49, 13],
    [33, 15],
    [93, 15],
    [53, 9],

    [59, 57],
    [63, 57],
    [65, 57],
    [67, 57],
    [69, 57],
    [71, 57]
]]