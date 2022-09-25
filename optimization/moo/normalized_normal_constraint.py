# Author(s): Qi FAN <qi dot fan at polytechnique dot edu>
#
# Description: Normalized Normal Constraint (NNC)
#
# Created at 25/09/2022

from optimization.moo.base_moo import BaseMOO

class NNC(BaseMOO):

    def __init__(self, inner_algo, obj_funcs, opt_type, const_funcs, const_types):
        super().__init__()
        pass