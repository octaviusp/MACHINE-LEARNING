class InvalidDataTypeException(Exception):
    def __init__(self, data_type: str):
        super().__init__(f"The value can't be saved as data type: {data_type}")