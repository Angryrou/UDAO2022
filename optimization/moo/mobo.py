# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Multi-Objective Bayesian Optimization (MOBO)
#
# Created at 25/09/2022

from optimization.moo.base_moo import BaseMOO

class MOBO(BaseMOO):

    def __init__(self, inner_algo, obj_funcs, opt_type, const_funcs, const_types):
        super().__init__()
        pass