"""ExFMECG model implementation used for training and inference."""

import contextlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from scripts.common.registry import registry
from scripts.models.base_model import BaseModel
from scripts.models.ecg_model.DBFN import ECGBackboneForXAttn
from scripts.models.ecg_model.phenomenonProject import PhenomenonProjector
from scripts.models.ked.clip_model import CLP_clinical, TQNModel


def _keep_frozen(module, mode=True):
    del mode
    return module


class AugCL(nn.Module):
    """Report-alignment loss retained from the ExFMECG training objective."""

    def __init__(self, temperature=0.05, loss_type="increase_dimension"):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type

    def forward(self, image_features, text_features, labels, used_for_concept=False):
        label_count = labels.shape[1]
        targets = torch.ones(
            (label_count, labels.shape[0], labels.shape[0]),
            device=image_features.device,
        )
        zero_mask = labels == 0
        for index in range(label_count):
            targets[index, zero_mask[:, index], :] = 0
            targets[index, :, zero_mask[:, index]] = 0

        image_logits = image_features @ text_features.T
        text_logits = image_logits.T
        temperature = 0.1 if used_for_concept else self.temperature

        if self.loss_type == "increase_dimension":
            scale = F.normalize(
                torch.ones(label_count, device=image_features.device),
                p=2,
                dim=0,
            ) * np.log(1 / temperature)
            scale = scale.view(-1, 1, 1)
            image_logits = (image_logits * scale).clamp(-100.0, 100.0)
            text_logits = (text_logits * scale).clamp(-100.0, 100.0)
            image_loss = F.kl_div(
                F.log_softmax(image_logits, dim=-1),
                targets,
                reduction="batchmean",
            )
            text_loss = F.kl_div(
                F.log_softmax(text_logits, dim=-1),
                targets,
                reduction="batchmean",
            )
        else:
            scale = np.log(1 / temperature)
            targets = targets.max(dim=0).values
            image_loss = F.kl_div(
                F.log_softmax(scale * image_logits, dim=-1),
                targets,
                reduction="sum",
            ) / image_logits.shape[1]
            text_loss = F.kl_div(
                F.log_softmax(scale * text_logits, dim=-1),
                targets,
                reduction="sum",
            ) / image_logits.shape[1]
        return (image_loss + text_loss) / (2 * image_features.shape[0])


@registry.register_model("exfmecg_v13")
class ExFMECGV13(BaseModel):
    """ExFMECG with 570 signal-derived and 739 binary concept outputs."""

    PRETRAINED_MODEL_CONFIG_DICT = {
        "v13": "config/model/exfmecg_v13.yaml",
    }
    DEFAULT_ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets"

    def __init__(
        self,
        bert_model_name,
        max_length,
        freeze_layers,
        unfreeze_layers,
        tqn_model_layers,
        freeze_vit=False,
        freeze_knowledge=False,
        max_txt_len=16,
        embed_dim=512,
        projector_hidden_size=1024,
        output_type="total",
        eval_dataset_type="sfp",
        evaluate_label_list=None,
        mode="train",
        report_max_length=128,
        concept_loss_weight=200.0,
        orth_loss_weight=0.0,
        clip_loss_weight=1.0,
        concept_loss_type="bce",
        v6_num_concepts=570,
        concept_matrix_root=None,
        concept_stats_dir=None,
        concept_routing=None,
        la_bce_tau=1.0,
        concept_w_bin=4.0,
        concept_w_reg=1.0,
        concept_w_cat=0.4,
        label_concept_runtime=None,
        label_neg_default_dts=None,
        use_renji_tasks=True,
        asset_root=None,
    ):
        super().__init__()
        self.mode = mode
        self.asset_root = self._resolve_root(asset_root)
        with (self.asset_root / "task_labels.json").open(encoding="utf-8") as handle:
            self.task_labels = json.load(handle)

        self.max_length = max_length
        self.max_txt_len = max_txt_len
        self.report_max_length = report_max_length
        self.embed_dim = embed_dim
        self.eval_dataset_type = eval_dataset_type
        self.evaluate_label_list = list(evaluate_label_list or [])
        self.freeze_knowledge = freeze_knowledge
        self.concept_loss_weight = float(concept_loss_weight)
        self.orth_loss_weight = float(orth_loss_weight)
        self.clip_loss_weight = float(clip_loss_weight)
        self.concept_loss_type = concept_loss_type

        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name,
            do_lower_case=True,
        )
        self.ecg_model = ECGBackboneForXAttn()
        self.knowledge_encoder = CLP_clinical(
            bert_model_name=bert_model_name,
            embed_dim=embed_dim,
            freeze_layers=freeze_layers,
            unfreeze_layers=unfreeze_layers,
        )
        self.tqn_model = TQNModel(
            embed_dim=embed_dim,
            num_layers=tqn_model_layers,
            output_type=output_type,
            mode=mode,
        )
        self.mlp_embed = nn.Sequential(
            nn.Linear(768, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if freeze_vit:
            for parameter in self.ecg_model.parameters():
                parameter.requires_grad = False
            self.ecg_model.eval()
            self.ecg_model.train = _keep_frozen.__get__(self.ecg_model, nn.Module)
        if freeze_knowledge:
            for parameter in self.knowledge_encoder.parameters():
                parameter.requires_grad = False
            self.knowledge_encoder.eval()
            self.knowledge_encoder.train = _keep_frozen.__get__(
                self.knowledge_encoder,
                nn.Module,
            )

        self.use_concept_classify = True
        self.use_v6_concept = True
        self.use_label_concepts = True
        self.cv6_w_bin = float(concept_w_bin)
        self.cv6_w_reg = float(concept_w_reg)
        self.cv6_w_cat = float(concept_w_cat)
        self._label_concept_runtime_path = label_concept_runtime
        self._label_neg_default_dts_cfg = label_neg_default_dts
        self._use_renji_tasks = bool(use_renji_tasks)
        self.load_v6_concept_config(
            concept_matrix_root,
            concept_stats_dir,
            n_concepts=v6_num_concepts,
            la_bce_tau=la_bce_tau,
            routing=concept_routing,
        )

        self.sfp_label_list = list(self.task_labels["sfph_ecg_phenotypes"])
        self.renji_ecg_list = list(self.task_labels["srh_ecg_phenotypes"])
        self.heedb_ecg_list = list(self.task_labels["mgh_ecg_phenotypes"])
        self.mimiciv_icd_list = [
            f"diagnosed with {label} at discharge"
            for label in self.task_labels["bidmc_diseases"]
        ]
        self.mimiciv_demo_list = list(self.task_labels["mortality_targets"])
        self.heedb_icd_list = list(self.task_labels["mgh_diseases"])
        self.renji_train_icd_list = [
            f"diagnosed with {label} at discharge"
            for label in self.task_labels["srh_diseases"]
        ]
        self.disease_pred_labels = list(
            self.task_labels["incident_disease_targets"]
        )

        for task_name in (
            "sfp",
            "renji_ecg",
            "heedb_ecg",
            "mimiciv_icd",
            "mimiciv_demo",
            "heedb_icd",
            "renji_train_icd",
        ):
            labels = getattr(self, f"{task_name}_label_list", None)
            if labels is None:
                labels = getattr(self, f"{task_name}_list")
            setattr(
                self,
                f"{task_name}_dict",
                {label: index for index, label in enumerate(labels)},
            )
        self.heedb_disease_pred_dict = {
            label: index for index, label in enumerate(self.disease_pred_labels)
        }
        self.mimiciv_disease_pred_dict = dict(self.heedb_disease_pred_dict)

        self.tp_mapping = {
            2: "sfp",
            4: "renji_ecg",
            5: "heedb_ecg",
            6: "heedb_ecg",
            10: "mimiciv_demo",
            11: "mimiciv_icd",
            35: "mimiciv_disease_pred",
            40: "heedb_disease_pred",
            61: "renji_train_icd",
            161: "heedb_icd",
        }
        self.label_groups = [("sfp", self.sfp_label_list)]
        if self._use_renji_tasks:
            self.label_groups.append(("renji_ecg", self.renji_ecg_list))
        self.label_groups.extend(
            [
                ("heedb_ecg", self.heedb_ecg_list),
                ("mimiciv_icd", self.mimiciv_icd_list),
                ("mimiciv_demo", self.mimiciv_demo_list),
                ("heedb_icd", self.heedb_icd_list),
            ]
        )
        if self._use_renji_tasks:
            self.label_groups.append(("renji_train_icd", self.renji_train_icd_list))
        self.label_groups.extend(
            [
                ("heedb_disease_pred", self.disease_pred_labels),
                ("mimiciv_disease_pred", self.disease_pred_labels),
            ]
        )

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.label_cl_loss = AugCL()
        self.concept_cl_loss = AugCL()
        self.llm_proj = nn.Sequential(
            nn.Linear(embed_dim, projector_hidden_size),
            nn.GELU(),
            nn.Linear(projector_hidden_size, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

        all_labels = [label for _, labels in self.label_groups for label in labels]
        with torch.no_grad():
            text_features = self.get_text_features(
                self.knowledge_encoder,
                all_labels,
                self.tokenizer,
                device="cpu",
                max_length=self.max_length,
            )
        self.register_buffer("all_label_features", text_features)
        self._cached_label_tokens = self.tokenizer(
            all_labels,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.group_info = [
            (task_name, len(labels)) for task_name, labels in self.label_groups
        ]
        self.log_vars_multitask = nn.Parameter(torch.zeros(len(self.label_groups)))
        self.register_buffer("datasize_weights", torch.ones(len(self.label_groups)))

    @classmethod
    def _resolve_root(cls, asset_root):
        if asset_root is None:
            return cls.DEFAULT_ASSET_ROOT
        root = Path(asset_root)
        if not root.is_absolute():
            root = Path(__file__).resolve().parents[2] / root
        return root

    def _resolve_asset(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        return path

    def load_v6_concept_config(
        self,
        matrix_root,
        stats_dir,
        n_concepts=570,
        la_bce_tau=1.0,
        routing=None,
    ):
        from scripts.datasets.datasets.concept_matrix_loader import ConceptMatrixLoader

        schema_dir = self.asset_root / "concepts" / "schema"
        stats_dir = (
            self._resolve_asset(stats_dir)
            if stats_dir
            else self.asset_root / "concepts" / "stats"
        )
        with (schema_dir / "concept_ids.json").open(encoding="utf-8") as handle:
            self.cv6_concept_ids = json.load(handle)
        with (schema_dir / "category_vocab.json").open(encoding="utf-8") as handle:
            category_vocab = json.load(handle)
        with (schema_dir / "binary_concept_ids.json").open(encoding="utf-8") as handle:
            binary_concept_ids = json.load(handle)

        if len(self.cv6_concept_ids) != n_concepts:
            raise ValueError("Concept schema does not match the configured count")
        self.cv6_n = n_concepts
        concept_index = {
            concept: index for index, concept in enumerate(self.cv6_concept_ids)
        }
        self.cv6_cat_cols = [
            concept_index[name] for name in category_vocab if name in concept_index
        ]
        self.cv6_cat_K = [
            len(category_vocab[name]) for name in category_vocab if name in concept_index
        ]

        loss_weight = np.load(schema_dir / "loss_weight.npy").astype("float32")
        prevalence = np.load(stats_dir / "pi_c.npy").astype("float64")
        center = np.load(stats_dir / "cont_center.npy").astype("float32")
        scale = np.clip(
            np.load(stats_dir / "cont_scale.npy"),
            1e-6,
            None,
        ).astype("float32")
        logit_shift = (
            la_bce_tau * (np.log(prevalence) - np.log1p(-prevalence))
        ).astype("float32")
        logit_shift[self.cv6_cat_cols] = 0.0

        runtime_path = self._resolve_asset(self._label_concept_runtime_path)
        with runtime_path.open(encoding="utf-8") as handle:
            runtime = json.load(handle)
        self.lc_l2c = {
            key.lower(): value for key, value in runtime["label_concept"].items()
        }
        self.lc_new_idx = {
            key: int(value) for key, value in runtime["new_col_index"].items()
        }
        self.lc_dual = {
            key: int(value) for key, value in runtime["dual_map"].items()
        }
        self.lc_head_binary = int(runtime["head_binary"])
        loss_weight = np.concatenate(
            [
                loss_weight,
                np.ones(self.lc_head_binary - n_concepts, dtype="float32"),
            ]
        )
        logit_shift = np.concatenate(
            [
                logit_shift,
                np.zeros(self.lc_head_binary - n_concepts, dtype="float32"),
            ]
        )
        self.register_buffer("cv6_loss_weight", torch.from_numpy(loss_weight))
        self.register_buffer("cv6_shift", torch.from_numpy(logit_shift))
        self.register_buffer("cv6_cont_center", torch.from_numpy(center))
        self.register_buffer("cv6_cont_scale", torch.from_numpy(scale))
        self.concept_projector = PhenomenonProjector(
            n_concepts=n_concepts,
            n_binary=self.lc_head_binary,
            cat_num_classes=self.cv6_cat_K,
        )

        self.cv6_routing = routing or {
            5: ("heedb", "study_id"),
            6: ("heedb", "study_id"),
            2: ("sfp", "study_id"),
        }
        matrix_root = Path(matrix_root) if matrix_root else None
        modalities = {"heedb"} | {
            modality for modality, _ in self.cv6_routing.values()
        }
        self.cv6_loaders = {
            modality: ConceptMatrixLoader(
                matrix_root / f"{modality}_matrix" if matrix_root else None,
                n_concepts,
            )
            for modality in sorted(modalities)
        }
        self._cv6_fb = self.cv6_loaders["heedb"]._fallback()
        routed_types = set(int(value) for value in self.cv6_routing)
        self.lc_pheno_dts = set(
            int(value)
            for value in (self._label_neg_default_dts_cfg or routed_types)
        )
        self.sfp_concept_list = binary_concept_ids

    def _gather_concept_targets(self, samples, to_device=True):
        dataset_types = samples["dataset_type"]
        if torch.is_tensor(dataset_types):
            dataset_types = dataset_types.tolist()
        rows = []
        for row_index, dataset_type in enumerate(dataset_types):
            route = self.cv6_routing.get(int(dataset_type))
            if route is None:
                rows.append(None)
                continue
            modality, id_field = route
            identifier = samples.get(id_field)
            identifier = identifier[row_index] if identifier is not None else None
            if torch.is_tensor(identifier):
                identifier = identifier.item()
            if isinstance(identifier, (int, float)):
                identifier = str(int(identifier))
            elif identifier is not None:
                identifier = str(identifier)
            loader = self.cv6_loaders.get(modality)
            rows.append(loader.fetch(identifier) if identifier is not None else None)

        output = {}
        for key, fallback in self._cv6_fb.items():
            tensor = torch.stack(
                [row[key] if row is not None else fallback for row in rows]
            )
            output[key] = tensor.to(self.device) if to_device else tensor
        return output

    def _augment_label_concept_targets(self, targets, samples):
        binary = targets["concept_binary"]
        mask = targets["concept_binary_mask"]
        if binary.shape[1] < self.lc_head_binary:
            width = self.lc_head_binary - binary.shape[1]
            binary = torch.cat(
                [binary, binary.new_zeros((binary.shape[0], width))],
                dim=1,
            )
            mask = torch.cat(
                [mask, mask.new_zeros((mask.shape[0], width))],
                dim=1,
            )

        dataset_types = samples["dataset_type"]
        if torch.is_tensor(dataset_types):
            dataset_types = dataset_types.tolist()
        for row, dataset_type in enumerate(dataset_types):
            if int(dataset_type) not in self.lc_pheno_dts:
                continue
            mask[row, self.cv6_n:] = 1
            labels = samples["label_list"][row]
            labels = labels.split("#") if isinstance(labels, str) else labels
            for label in labels or []:
                for concept in self.lc_l2c.get(str(label).strip().lower(), ()):
                    new_index = self.lc_new_idx.get(concept)
                    if new_index is not None:
                        binary[row, new_index] = 1
                    else:
                        existing_index = self.lc_dual.get(concept)
                        if existing_index is not None:
                            binary[row, existing_index] = 1
                            mask[row, existing_index] = 1
        targets["concept_binary"] = binary
        targets["concept_binary_mask"] = mask
        return targets

    def _concept_binary_loss(self, logits, targets, mask):
        targets = targets.clamp(min=0).float()
        mask = mask.float()
        shift = self.cv6_shift.view(1, -1) if self.concept_loss_type == "la_bce" else 0.0
        loss = F.binary_cross_entropy_with_logits(
            logits + shift,
            targets,
            reduction="none",
        )
        weights = self.cv6_loss_weight.view(1, -1)
        return (loss * mask * weights).sum() / (mask * weights).sum().clamp_min(1.0)

    def _concept_regression_loss(self, predictions, targets, mask):
        valid = (mask > 0) & torch.isfinite(targets)
        if not valid.any():
            return predictions.sum() * 0.0
        targets = (
            (targets - self.cv6_cont_center.view(1, -1))
            / self.cv6_cont_scale.view(1, -1)
        ).clamp(-10.0, 10.0)
        return F.smooth_l1_loss(predictions[valid], targets[valid], reduction="mean")

    def _concept_categorical_loss(self, logits, targets, mask):
        total = None
        count = 0
        for index, column in enumerate(self.cv6_cat_cols):
            classes = int(self.cv6_cat_K[index])
            target = targets[:, column].long()
            valid = (mask[:, column] > 0) & (target >= 0) & (target < classes)
            if valid.any():
                term = F.cross_entropy(logits[index][valid], target[valid], reduction="sum")
                total = term if total is None else total + term
                count += int(valid.sum())
        if total is None:
            return logits[0].sum() * 0.0
        return total / count

    def _task_label_features(self, device):
        if self.freeze_knowledge:
            features = self.mlp_embed(self.all_label_features.to(device))
        else:
            features = self.mlp_embed(
                self.get_text_features_from_tokens(
                    self.knowledge_encoder,
                    self._cached_label_tokens,
                    device,
                )
            )
        output = {}
        start = 0
        for task_name, length in self.group_info:
            output[task_name] = features[start:start + length]
            start += length
        return output

    def _build_task_targets(self, samples, label_features, device):
        batch_size = samples["dataset_type"].shape[0]
        matrices = {
            task_name: torch.full(
                (batch_size, len(getattr(self, f"{task_name}_dict"))),
                -1,
                device=device,
            )
            for task_name in label_features
        }
        selected_rows = defaultdict(list)
        dataset_types = samples["dataset_type"]
        labels = samples["label_list"]
        evaluable = samples.get("evaluable_labels", [""] * batch_size)
        partially_observed = {"heedb_disease_pred", "mimiciv_disease_pred"}

        for dataset_type, task_name in self.tp_mapping.items():
            rows = (dataset_types == dataset_type).nonzero(as_tuple=True)[0]
            if rows.numel() == 0 or task_name not in matrices:
                continue
            label_index = getattr(self, f"{task_name}_dict")
            matrix = matrices[task_name]
            if task_name in partially_observed:
                for row in rows.tolist():
                    positive = str(labels[row]).split("#") if labels[row] else []
                    if evaluable[row]:
                        for label in str(evaluable[row]).split("#"):
                            if label in label_index:
                                matrix[row, label_index[label]] = 0
                    else:
                        matrix[row] = 0
                    for label in positive:
                        if label in label_index:
                            matrix[row, label_index[label]] = 1
                selected_rows[task_name].extend(rows.tolist())
                continue

            positive_rows = []
            positive_columns = []
            for row in rows.tolist():
                positive = str(labels[row]).split("#") if labels[row] else []
                for label in positive:
                    if label in label_index:
                        positive_rows.append(row)
                        positive_columns.append(label_index[label])
            if positive_rows:
                matrix[rows] = 0
                matrix[positive_rows, positive_columns] = 1
                selected_rows[task_name].extend(rows.tolist())
        return matrices, selected_rows

    @staticmethod
    def _report_label_matrix(label_lists, device):
        unique_labels = sorted({label for labels in label_lists for label in labels})
        index = {label: position for position, label in enumerate(unique_labels)}
        matrix = torch.zeros(
            len(label_lists),
            len(unique_labels),
            dtype=torch.int8,
            device=device,
        )
        for row, labels in enumerate(label_lists):
            if labels:
                matrix[row, [index[label] for label in labels]] = 1
        return matrix

    def forward(self, samples):
        signal = samples["ecg"].float()
        metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
        age = torch.as_tensor(samples["age"], device=signal.device, dtype=metadata_dtype).view(-1, 1)
        gender = torch.as_tensor(
            samples["gender"],
            device=signal.device,
            dtype=metadata_dtype,
        ).view(-1, 1)
        metadata = torch.cat([age, gender], dim=1)

        concept_future = None
        if getattr(self, "_concept_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor

            self._concept_pool = ThreadPoolExecutor(max_workers=1)
        concept_future = self._concept_pool.submit(
            self._gather_concept_targets,
            samples,
            False,
        )

        with self.maybe_autocast():
            ecg_features = self.ecg_model(signal, metadata)
            concept_grid = F.avg_pool2d(ecg_features, kernel_size=4, stride=4).flatten(1)
            ecg_queries = ecg_features.transpose(1, 2)
            pooled_ecg = ecg_queries.mean(1)
            label_features = self._task_label_features(signal.device)

        target_matrices, selected_rows = self._build_task_targets(
            samples,
            label_features,
            signal.device,
        )
        losses = {}
        for task_name, text_features in label_features.items():
            rows = selected_rows.get(task_name, [])
            if not rows:
                losses[f"{task_name}_loss"] = 0
                continue
            logits = self.llm_proj(self.tqn_model(ecg_queries[rows], text_features))
            losses[f"{task_name}_loss"] = self.ce_loss(
                logits.reshape(-1, 2),
                target_matrices[task_name][rows].reshape(-1),
            )

        concept_targets = {
            key: value.to(concept_grid.device)
            for key, value in concept_future.result().items()
        }
        concept_targets = self._augment_label_concept_targets(
            concept_targets,
            samples,
        )
        binary_logits = self.concept_projector.head_binary(concept_grid)
        regression_predictions = self.concept_projector.head_regression(concept_grid)
        categorical_logits = [
            head(concept_grid) for head in self.concept_projector.head_categorical
        ]
        binary_loss = self._concept_binary_loss(
            binary_logits,
            concept_targets["concept_binary"],
            concept_targets["concept_binary_mask"],
        )
        regression_loss = self._concept_regression_loss(
            regression_predictions,
            concept_targets["concept_continuous"],
            concept_targets["concept_regression_mask"],
        )
        categorical_loss = self._concept_categorical_loss(
            categorical_logits,
            concept_targets["concept_categorical"],
            concept_targets["concept_categorical_mask"],
        )
        concept_loss = (
            self.cv6_w_bin * binary_loss
            + self.cv6_w_reg * regression_loss
            + self.cv6_w_cat * categorical_loss
        )

        label_lists = [
            str(value).split("#") if value else [] for value in samples["label_list"]
        ]
        with torch.no_grad():
            report_bert = self.get_text_features(
                self.knowledge_encoder,
                samples["report"],
                self.tokenizer,
                signal.device,
                max_length=self.report_max_length,
            )
        report_features = self.mlp_embed(report_bert)
        report_targets = self._report_label_matrix(label_lists, signal.device)
        alignment_loss = self.label_cl_loss(
            pooled_ecg,
            report_features,
            report_targets,
        )

        task_weights = {
            "sfp_loss": len(self.sfp_dict),
            "renji_ecg_loss": len(self.renji_ecg_list),
            "heedb_ecg_loss": len(self.heedb_ecg_list),
            "mimiciv_icd_loss": len(self.mimiciv_icd_list) * 2,
            "mimiciv_demo_loss": len(self.mimiciv_demo_list),
            "heedb_icd_loss": len(self.heedb_icd_list) * 2,
            "renji_train_icd_loss": len(self.renji_train_icd_list) * 2,
            "heedb_disease_pred_loss": len(self.disease_pred_labels),
            "mimiciv_disease_pred_loss": len(self.disease_pred_labels),
        }
        total_loss = signal.new_zeros(())
        for name, loss in losses.items():
            if torch.is_tensor(loss):
                total_loss = total_loss + loss * task_weights[name]
        total_loss = total_loss + self.concept_loss_weight * concept_loss
        total_loss = total_loss + self.clip_loss_weight * alignment_loss

        if self.orth_loss_weight > 0:
            weights = self.concept_projector.head_binary.weight
            identity = torch.eye(weights.shape[0], device=weights.device)
            orthogonality_loss = torch.linalg.matrix_norm(
                weights @ weights.T - identity,
                ord="fro",
            ).square()
            total_loss = total_loss + self.orth_loss_weight * orthogonality_loss
        else:
            orthogonality_loss = total_loss.new_zeros(())

        losses.update(
            {
                "loss": total_loss,
                "clip_loss": alignment_loss,
                "concept_loss": concept_loss,
                "concept_binary_loss": binary_loss.detach(),
                "concept_reg_loss": regression_loss.detach(),
                "concept_cat_loss": categorical_loss.detach(),
                "orth_loss": orthogonality_loss,
            }
        )
        return losses

    @torch.no_grad()
    def predict_concepts(self, samples):
        signal = samples["ecg"].float()
        metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
        age = torch.as_tensor(
            samples["age"], device=signal.device, dtype=metadata_dtype
        ).view(-1, 1)
        gender = torch.as_tensor(
            samples["gender"],
            device=signal.device,
            dtype=metadata_dtype,
        ).view(-1, 1)
        with self.maybe_autocast():
            features = self.ecg_model(signal, torch.cat([age, gender], dim=1))
            grid = F.avg_pool2d(features, kernel_size=4, stride=4).flatten(1)
            binary = torch.sigmoid(self.concept_projector.head_binary(grid))
            continuous = self.concept_projector.head_regression(grid)
            categorical = [
                torch.softmax(head(grid), dim=-1)
                for head in self.concept_projector.head_categorical
            ]
        return {
            "binary": binary.float(),
            "continuous": continuous.float(),
            "categorical": [output.float() for output in categorical],
        }

    @torch.no_grad()
    def generate(self, samples, queries=None):
        signal = samples["ecg"].float()
        metadata_dtype = torch.float16 if signal.is_cuda else torch.float32
        age = torch.as_tensor(samples["age"], device=signal.device, dtype=metadata_dtype).view(-1, 1)
        gender = torch.as_tensor(
            samples["gender"],
            device=signal.device,
            dtype=metadata_dtype,
        ).view(-1, 1)
        with self.maybe_autocast():
            features = self.ecg_model(signal, torch.cat([age, gender], dim=1))
            features = features.transpose(1, 2)

        task_labels = {name: labels for name, labels in self.label_groups}
        default_queries = list(queries or self.evaluate_label_list)
        task_labels["default"] = default_queries
        max_labels = max(len(labels) for labels in task_labels.values())
        predictions = torch.full(
            (signal.shape[0], max_labels),
            -1.0,
            device=signal.device,
        )
        dataset_types = samples["dataset_type"]
        if not torch.is_tensor(dataset_types):
            dataset_types = torch.tensor(dataset_types, device=signal.device)

        for dataset_type in torch.unique(dataset_types):
            rows = (dataset_types == dataset_type).nonzero(as_tuple=True)[0]
            task_name = self.tp_mapping.get(int(dataset_type.item()), "default")
            current_queries = task_labels.get(task_name, default_queries)
            if default_queries and len(default_queries) < len(current_queries):
                current_queries = default_queries
            if not current_queries:
                raise ValueError("At least one query is required for inference")
            text_features = self.get_text_features(
                self.knowledge_encoder,
                current_queries,
                self.tokenizer,
                signal.device,
                self.max_length,
            )
            with self.maybe_autocast():
                logits = self.llm_proj(
                    self.tqn_model(
                        features[rows],
                        self.mlp_embed(text_features),
                    )
                )
            probabilities = torch.softmax(logits, dim=-1)[..., 1]
            predictions[rows, :len(current_queries)] = probabilities.float()
        return predictions

    def materialize_dynamic_modules(self):
        """Instantiate input-dependent ECA convolutions before checkpoint loading."""
        device = self.device
        with torch.no_grad():
            self.ecg_model(
                torch.zeros(1, 12, 1024, device=device),
                torch.zeros(1, 2, device=device),
            )

    @staticmethod
    def get_text_features(model, text_list, tokenizer, device, max_length):
        tokens = tokenizer(
            list(text_list),
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device=device)
        return model.encode_text(tokens)

    @staticmethod
    def get_text_features_from_tokens(model, cached_tokens, device):
        tokens = {name: value.to(device) for name, value in cached_tokens.items()}
        return model.encode_text(tokens)

    def maybe_autocast(self, dtype=torch.float16):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    @classmethod
    def from_config(cls, cfg):
        model = cls(
            bert_model_name=cfg.get(
                "bert_model_name",
                "emilyalsentzer/Bio_ClinicalBERT",
            ),
            max_length=cfg.get("max_txt_len", 16),
            freeze_layers=cfg.get("freeze_layers", list(range(11))),
            unfreeze_layers=cfg.get("unfreeze_layers", [9, 10, 11]),
            tqn_model_layers=cfg.get("tqn_model_layers", 12),
            freeze_vit=cfg.get("freeze_vit", False),
            freeze_knowledge=cfg.get("freeze_knowledge", False),
            max_txt_len=cfg.get("max_txt_len", 16),
            embed_dim=cfg.get("embed_dim", 512),
            output_type=cfg.get("output_type", "total"),
            eval_dataset_type=cfg.get("eval_dataset_type", "sfp"),
            evaluate_label_list=cfg.get("evaluate_label_list", []),
            mode=cfg.get("mode", "train"),
            report_max_length=cfg.get("report_max_length", 128),
            concept_loss_weight=cfg.get("concept_loss_weight", 200.0),
            orth_loss_weight=cfg.get("orth_loss_weight", 0.0),
            clip_loss_weight=cfg.get("clip_loss_weight", 1.0),
            concept_loss_type=cfg.get("concept_loss_type", "bce"),
            v6_num_concepts=cfg.get("v6_num_concepts", 570),
            concept_matrix_root=cfg.get("concept_matrix_root"),
            concept_stats_dir=cfg.get("concept_stats_dir"),
            concept_routing={
                int(key): tuple(value)
                for key, value in cfg.get("concept_routing", {}).items()
            } or None,
            la_bce_tau=cfg.get("la_bce_tau", 1.0),
            concept_w_bin=cfg.get("concept_w_bin", 4.0),
            concept_w_reg=cfg.get("concept_w_reg", 1.0),
            concept_w_cat=cfg.get("concept_w_cat", 0.4),
            label_concept_runtime=cfg.get(
                "label_concept_runtime",
                "assets/concepts/label_concept_runtime.json",
            ),
            label_neg_default_dts=cfg.get("label_neg_default_dts", [2, 4, 5, 6]),
            use_renji_tasks=cfg.get("use_renji_tasks", True),
            asset_root=cfg.get("asset_root"),
        )
        if cfg.get("freeze_unused_params", True):
            for name, parameter in model.named_parameters():
                if ".aspp." in name or name.endswith("logit_scale") or "log_vars_multitask" in name:
                    parameter.requires_grad = False
        model.load_checkpoint_from_config(cfg)
        return model
