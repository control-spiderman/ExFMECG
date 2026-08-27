from scripts.common.registry import registry
from scripts.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from scripts.datasets.datasets.exfmecg_dataset import ExFMECGManifestDataset


@registry.register_builder("exfmecg")
class ExFMECGBuilder(BaseDatasetBuilder):
    def build_datasets(self):
        datasets = {
            "train": ExFMECGManifestDataset(
                manifest=self.config.train_manifest,
                ecg_root=self.config.get("ecg_root", "."),
            )
        }
        if self.config.get("validation_manifest"):
            datasets["val"] = ExFMECGManifestDataset(
                manifest=self.config.validation_manifest,
                ecg_root=self.config.get("ecg_root", "."),
            )
        return datasets
