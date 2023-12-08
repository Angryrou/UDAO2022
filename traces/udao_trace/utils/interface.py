from enum import Enum


class VarTypes(Enum):
    INT = "int"
    BOOL = "bool"
    CATEGORY = "category"
    FLOAT = "float"


class ScaleTypes(Enum):
    LOG = "log"
    LINEAR = "linear"