class NzPreprocessor:
    """
    Base class for data preprocessing steps in neurozip.
    Extend this class to implement custom preprocessing logic.

    Example:
        class MyPreprocessor(NzPreprocessor):
            def process(self, data):
                # custom logic
                return data
    """
    def __init__(self):
        pass
