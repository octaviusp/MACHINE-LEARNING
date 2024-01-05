class InvalidAdditionException(Exception):
    def __init__(self, type_1: str, type_2: str):
        super().__init__(f"The values types must be the same to add, you're trying: {type_1} + {type_2}")