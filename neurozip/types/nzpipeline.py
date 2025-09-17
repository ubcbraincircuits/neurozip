class NzPipeline:
    """
    Pipeline for chaining multiple preprocessing steps.
    Each step should be a callable that takes and returns data.

    Args:
        steps (list): List of preprocessing callables to apply in sequence.

    Example:
        pipeline = NzPipeline([step1, step2])
        processed = pipeline.run(data)
    """

    def __init__(self, steps):
        self.steps = steps
