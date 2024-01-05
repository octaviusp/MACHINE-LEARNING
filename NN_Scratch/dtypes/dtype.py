#from error.invalid_operation_overflow import Overflow


class DTYPE():

    """
     Base class for data type.
    """

    def __init__(self, name: str, size_bits: int, class_type: any):
        self.name = name
        self.size_bits = size_bits
        self.class_type = class_type
