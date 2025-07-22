class CouldNotSearchQuery(Exception):
    def __init__(self, message, *args, **kwargs):
        super().__init__(message)


class CouldNotReadUrl(Exception):
    def __init__(self, message, *args, **kwargs):
        super().__init__(message)
