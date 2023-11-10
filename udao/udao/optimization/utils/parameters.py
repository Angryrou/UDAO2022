UCB_BETA = 0.2


class VarTypes:
    INTEGER = 1
    BOOL = 2
    ENUM = 3
    FLOAT = 4
    ALL_TYPES = {INTEGER, BOOL, ENUM, FLOAT}

    @staticmethod
    def str2type(s: str) -> int:
        type_dict = {"INTEGER": 1, "BOOL": 2, "ENUM": 3, "FLOAT": 4}
        try:
            return type_dict[s]
        except KeyError:
            raise Exception(f"VarType does not support {s}")
