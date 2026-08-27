"""Model registrations for the ExFMECG release."""

from scripts.models.base_model import BaseModel
from scripts.models.exfmecg_v13 import ExFMECGV13

__all__ = ["BaseModel", "ExFMECGV13"]
