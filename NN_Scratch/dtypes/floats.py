from .dtype import DTYPE
"""from ..error.exceptions import (
    InvalidAdditionException,
    InvalidDataTypeException
)"""

import numpy as np

class FLOAT(DTYPE):
    def __init__(self, value: float, class_name: str, class_bits: int, class_type: any):
        super(FLOAT, self).__init__(class_name, class_bits, class_type)
        self.value = self.class_type(value)
        self.type = self.class_type

    def __repr__(self) -> int:
        return self.value
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __add__(self, other: DTYPE):
        if type(other.value) is not self.type:
            raise Exception(f"Both numbers must be {self.type} or {other.type}")
        return self.class_type(self.value + other.value)
    
class FLOAT_16(FLOAT):
    def __init__(self, value: float):
        super(FLOAT_16, self).__init__(value, "float16", 16, np.float16)

class FLOAT_32(FLOAT):
    def __init__(self, value: float):
        super(FLOAT_32, self).__init__(value, "float32", 16, np.float32)

class FLOAT_64(FLOAT):
    def __init__(self, value: float):
        super(FLOAT_64, self).__init__(value, "float64", 16, np.float64)
