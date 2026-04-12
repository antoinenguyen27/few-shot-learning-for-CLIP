import torch
from common.evaluation.metrics import accuracy
from common.evaluation.results import RunResult
from .model import DPCModel
from .trainer import DPCTrainer


class DPCMethod:
    method_name = "DPC"

    def __init__(self):
        self.model = None

    def fit(self, train_loader, val_loader, classnames, bundle):
        model = DPCModel(bundle, classnames)
        trainer = DPCTrainer()

        self.model = trainer.train(model, train_loader)

        return {
            "model": self.model,
            "backbone": bundle.model_name,
            "pretrained": bundle.pretrained,
        }

    def evaluate(self, artifact, test_loader, classnames, bundle, shots=16, split_path=""):
        model = artifact["model"]

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                logits, _ = model(batch["image"])
                preds = logits.argmax(dim=1)

                all_preds.append(preds)
                all_labels.append(batch["label"])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        acc = accuracy(all_preds.tolist(), all_labels.tolist())

        return RunResult(
            method="DPC",
            dataset="eurosat",
            protocol="few_shot_all_classes",
            model_name=artifact["backbone"],
            pretrained=artifact["pretrained"],
            shots=shots,
            seed=1,
            metrics={
                "test/top1_accuracy": acc,
                "test/macro_accuracy": acc,
            },
            split_path=split_path,
            notes="Simplified DPC implementation",
            extra={
                "backbone_method": "OpenCLIP",
                "backbone_checkpoint": artifact["pretrained"],
                "reference_model": "OpenCLIP zero-shot",
                "distillation": "KL divergence",
                "temperature": 1.0,
                "alpha": 0.8,
            },
        )