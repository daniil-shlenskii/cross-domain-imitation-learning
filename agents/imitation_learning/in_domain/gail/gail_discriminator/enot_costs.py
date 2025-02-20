from utils.custom_types import DataType


def gwil_cost(source: DataType, target: DataType):
    source_dist = (source["observations_next"] - source["observations"])**2
    target_dist = (target["observations_next"] - target["observations"])**2
    cost = (source_dist.mean(-1) - target_dist.mean(-1))**2
    return cost
