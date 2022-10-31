# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description: TODO
#
# Created at 10/29/22
from utils.common import JsonUtils

AWS_COST_CPU_HOUR_RATIO = 0.052624
AWS_COST_MEM_HOUR_RATIO = 0.0057785  # for GB*H


def get_cloud_cost(lat, mem, cores, nexec):
    cpu_hour = (nexec + 1) * cores * lat / 3600
    mem_hour = (nexec + 1) * mem * lat / 3600
    cost = cpu_hour * AWS_COST_CPU_HOUR_RATIO + mem_hour * AWS_COST_MEM_HOUR_RATIO
    return cost
