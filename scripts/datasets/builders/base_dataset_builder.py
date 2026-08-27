"""Dataset-builder interface retained from the original training framework."""


class BaseDatasetBuilder:
    def __init__(self, cfg):
        self.config = cfg

    def build_datasets(self):
        raise NotImplementedError
