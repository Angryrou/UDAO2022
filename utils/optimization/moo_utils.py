# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#            Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: common methods used in MOO
# reuse some code in VLDB2022

# Created at 15/09/2022

import numpy as np
from matplotlib import pyplot as plt

from utils.parameters import VarTypes


# a quite efficient way to get the indexes of pareto points
# https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs, return_mask=True):
    ## reuse code in VLDB2022
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def _summarize_ret(po_obj_list, po_var_list):
    ## reuse code in VLDB2022
    assert len(po_obj_list) == len(po_var_list)
    if len(po_obj_list) == 0:
        return None, None
    elif len(po_obj_list) == 1:
        return np.array(po_obj_list), np.array(po_var_list)
    else:
        po_objs_cand = np.array(po_obj_list)
        po_vars_cand = np.array(po_var_list)
        po_inds = is_pareto_efficient(po_objs_cand)
        po_objs = po_objs_cand[po_inds]
        po_vars = po_vars_cand[po_inds]
        return po_objs, po_vars

# generate even weights for 2D and 3D
def even_weights(stepsize, m):
    if m == 2:
        w1 = np.hstack([np.arange(0, 1, stepsize), 1])
        w2 = 1 - w1
        ws_pairs = [[w1, w2] for w1, w2 in zip(w1, w2)]

    elif m == 3:
        w_steps = np.linspace(0, 1, num=1 / stepsize + 1, endpoint=True)
        for i, w in enumerate(w_steps):
            # use round to avoid case of floating point limitations in Python
            # the limitation: 1- 0.9 = 0.09999999999998 rather than 0.1
            other_ws_range = round((1 - w), 10)
            w2 = np.linspace(0, other_ws_range, num=round(other_ws_range/stepsize + 1), endpoint=True)
            w3 = other_ws_range - w2
            num = w2.shape[0]
            w1 = np.array([w] * num)
            ws = np.hstack([w1.reshape([num, 1]), w2.reshape([num, 1]), w3.reshape([num, 1])])
            if i == 0:
                ws_pairs = ws
            else:
                ws_pairs = np.vstack([ws_pairs, ws])

    assert all(np.sum(ws_pairs, axis=1) == 1)
    return ws_pairs

# common functions used in moo
def _get_direction(opt_type, obj_index):
    if opt_type[obj_index] == "MIN":
        return 1
    else:
        return -1


def plot_po(po, n_obj=2):
    # po: ndarray (n_solutions * n_objs)
    ## for 2D
    po_obj1 = po[:, 0]
    po_obj2 = po[:, 1]

    fig, ax = plt.subplots()
    ax.scatter(po_obj1, po_obj2, marker='o', color="blue")

    ax.set_xlabel('Obj 1')
    ax.set_ylabel('Obj 2')

    plt.tight_layout()
    plt.show()

# generate training inputs for GPR, reuse code in RandomSampler solver
def get_training_input(var_types, var_ranges, n_samples):
    '''
    generate samples of variables
    :param var_ranges: array (n_vars,), lower and upper var_ranges of variables(non-ENUM), and values of ENUM variables
    :param var_types: list, type of each variable
    :param n_samples: int, the number of input samples to train GPR models
    :return: array, variables (n_samples * n_vars)
    '''
    n_vars = var_ranges.shape[0]
    x = np.zeros([n_samples, n_vars])
    np.random.seed(0)
    for i, values in enumerate(var_ranges):
        upper, lower = values[1], values[0]
        if (lower - upper) > 0:
            raise Exception(f"ERROR: the lower bound of variable {i} is greater than its upper bound!")

        # randomly sample n_samples within the range
        if var_types[i] == VarTypes.FLOAT:
            x[:, i] = rand_float(lower, upper, n_samples)
        elif var_types[i] == VarTypes.INTEGER or var_types[i] == VarTypes.BOOL:
            x[:, i] = np.random.randint(lower, upper + 1, size=n_samples)
        elif var_types[i] == VarTypes.ENUM:
            inds = np.random.randint(0, len(values), size=n_samples)
            x[:, i] = np.array(values)[inds]
        else:
            raise Exception(f"Variable type {var_types[i]} is not supported!")
    return x

def rand_float(lower, upper, n_samples):
    '''
    generate n_samples random float values within the lower and upper var_ranges
    :param lower: int, lower bound
    :param upper: int upper bound
    :param n_samples: int, the number of samples
    :return: ndarray(n_samples, ), n_samples random float
    '''
    if lower > upper:
        return None
    else:
        scale = upper - lower
        out = np.random.rand(n_samples) * scale + lower
        return out