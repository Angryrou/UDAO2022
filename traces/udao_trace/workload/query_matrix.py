import numpy as np


class QueryMatrix:

    def __init__(self, templates: list[str], n_data_per_template: int, seed: int = 42):
        np.random.seed(seed)
        queries = np.tile(templates, [n_data_per_template, 1])
        self.queries = np.apply_along_axis(np.random.permutation, axis=1, arr=queries).flatten()
        self.total = len(self.queries)
        self.n_templates = len(templates)
        self.n_data_per_template = n_data_per_template

    def get_query_as_template_variant(self, i: int) -> (str, int):
        template = self.queries[i]
        variant = (i // self.n_templates) + 1 # variant indexed from 1, such q1-1, q1-2, q1-3, ...
        return template, variant