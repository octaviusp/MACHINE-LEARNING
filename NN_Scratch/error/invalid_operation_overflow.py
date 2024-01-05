class Overflow(Exception):
    def __init__(self, data_type: str, data_size: int):
        super().__init__(f"Overflow ocurred. Max size on {data_type} is {data_size} bits")