import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt

from .parameters import VarTypes


class Points:
    def __init__(self, objs: np.ndarray, vars: Optional[np.ndarray] = None) -> None:
        """
        # Docstring in numpy format

        Parameters
        ----------
        objs : np.ndarray(n_objs,)
            objective values
        vars :np.ndarray(1, n_vars), default=None
            variable values, by default None
        """
        self.objs = objs
        self.vars = vars
        self.n_objs = objs.shape[0]


class Rectangles:
    def __init__(self, utopia: Points, nadir: Points) -> None:
        """

        Parameters
        ----------
        utopia : Points
            utopia point
        nadir : Points
            nadir point
        """

        self.upper_bounds = nadir.objs
        self.lower_bounds = utopia.objs
        self.n_objs = nadir.objs.shape[0]
        self.volume = self.cal_volume(nadir.objs, utopia.objs)
        self.neg_vol = -self.volume
        self.utopia = utopia
        self.nadir = nadir

    def cal_volume(self, upper_bounds: np.ndarray, lower_bounds: np.ndarray) -> float:
        """
        Calculate the volume of the hyper_rectangle

        Parameters
        ----------
        upper_bounds : np.ndarray(n_objs,)
            upper bounds of the hyper_rectangle
        lower_bounds : np.ndarray(n_objs,)
            lower bounds of the hyper_rectangle

        Returns
        -------
        float
            volume of the hyper_rectangle
        """
        volume = np.abs(np.prod(upper_bounds - lower_bounds))
        return volume

    # Override the `__lt__()` function to make `Rectangles`
    # class work with min-heap (referred from VLDB2022)
    def __lt__(self, other: "Rectangles") -> bool:
        return self.neg_vol < other.neg_vol


# a quite efficient way to get the indexes of pareto points
# https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
    ## reuse code in VLDB2022
    """
    Find the pareto-efficient points

    Parameters
    ----------
    costs : np.ndarray
        An (n_points, n_costs) array
    return_mask : bool, default=True
        True to return a mask

    Returns
    -------
    np.ndarray
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


def _summarize_ret(
    po_obj_list: Sequence, po_var_list: Sequence
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return the pareto-optimal objectives and variables

    Parameters
    ----------
    po_obj_list: Sequence
        List of objective values
    po_var_list : _type_
        List of variable values

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        Pareto-optimal objectives and variables
    """
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


# generate even weights for 2d and 3D
def even_weights(stepsize: float, m: int) -> np.ndarray:
    ws_pairs = np.array([])
    if m == 2:
        w1 = np.hstack([np.arange(0, 1, stepsize), 1])
        w2 = 1 - w1
        ws_pairs = np.array([[w1, w2] for w1, w2 in zip(w1, w2)])

    elif m == 3:
        w_steps = np.linspace(0, 1, num=int(1 / stepsize) + 1, endpoint=True)
        for i, w in enumerate(w_steps):
            # use round to avoid case of floating point limitations in Python
            # the limitation: 1- 0.9 = 0.09999999999998 rather than 0.1
            other_ws_range = round((1 - w), 10)
            w2 = np.linspace(
                0,
                other_ws_range,
                num=round(other_ws_range / stepsize + 1),
                endpoint=True,
            )
            w3 = other_ws_range - w2
            num = w2.shape[0]
            w1 = np.array([w] * num)
            ws = np.hstack(
                [w1.reshape([num, 1]), w2.reshape([num, 1]), w3.reshape([num, 1])]
            )
            if i == 0:
                ws_pairs = ws
            else:
                ws_pairs = np.vstack([ws_pairs, ws])
    else:
        raise Exception(f"{m} objectives are not supported.")

    assert all(np.round(np.sum(ws_pairs, axis=1), 10) == 1)
    return np.array(ws_pairs)


# common functions used in moo
def _get_direction(opt_type: Sequence, obj_index: int) -> int:
    """Get gradient direction from optimization type"""
    if opt_type[obj_index] == "MIN":
        return 1
    else:
        return -1


def plot_po(po: np.ndarray, n_obj: int = 2, title: str = "pf_ap") -> None:
    """Plot pareto-optimal solutions"""
    # po: ndarray (n_solutions * n_objs)
    ## for 2d
    if n_obj == 2:
        po_obj1 = po[:, 0]
        po_obj2 = po[:, 1]

        fig, ax = plt.subplots()
        ax.scatter(po_obj1, po_obj2, marker="o", color="blue")
        ax.plot(po_obj1, po_obj2, color="blue")

        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")

        ax.set_title(title)

    elif n_obj == 3:
        po_obj1 = po[:, 0]
        po_obj2 = po[:, 1]
        po_obj3 = po[:, 2]

        plt.figure()
        ax = plt.axes(projection="3d")

        # ax.plot_trisurf(po_obj1, po_obj2, po_obj3, antialiased=True)
        # ax.plot_surface(po_obj1, po_obj2, po_obj3)
        ax.scatter3D(po_obj1, po_obj2, po_obj3, color="blue")

        ax.set_xlabel("Obj 1")
        ax.set_ylabel("Obj 2")
        ax.set_zlabel("Obj 3")

    else:
        raise Exception(
            f"{n_obj} objectives are not supported in the code repository for now!"
        )

    plt.tight_layout()
    plt.show()


# generate training inputs for GPR, reuse code in RandomSampler solver
def get_training_input(
    var_types: List[VarTypes], var_ranges: np.ndarray, n_samples: int
) -> np.ndarray:
    """
    Generate samples of variables (for the unconstrained scenario)
    Parameters
    ----------
    var_types : List[VarTypes]
        List of variable types
    var_ranges : np.ndarray
        lower and upper bounds of variables(non-ENUM),
        all available values for ENUM variables
    n_samples : int
        Number of input samples to train GPR models
    Returns
    -------
    np.ndarray
        Variables (n_samples * n_vars)
    """
    n_vars = var_ranges.shape[0]
    x = np.zeros([n_samples, n_vars])
    np.random.seed(0)
    for i, values in enumerate(var_ranges):
        upper, lower = values[1], values[0]
        if (lower - upper) > 0:
            raise Exception(
                f"ERROR: the lower bound of variable {i} "
                "is greater than its upper bound!"
            )

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


def rand_float(lower: float, upper: float, n_samples: int) -> np.ndarray | None:
    """
    generate n_samples random float values within the lower and upper var_ranges
    :param lower: int, lower bound
    :param upper: int upper bound
    :param n_samples: int, the number of samples
    :return: ndarray(n_samples, ), n_samples random float
    """
    if lower > upper:
        return None
    else:
        scale = upper - lower
        out = np.random.rand(n_samples) * scale + lower
        return out


def save_results(
    path: str, results: np.ndarray, wl_id: str, mode: str = "data"
) -> None:
    file_path = path + f"jobId_{wl_id}/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    np.savetxt(f"{file_path}/{mode}.txt", results)
