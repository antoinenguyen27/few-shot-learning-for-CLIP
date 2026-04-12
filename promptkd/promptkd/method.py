
import torch
import torch.nn.functional as F
import open_clip
from common.methods import MethodArtifact
from common.models.openclip import (
    build_zero_shot_classifier,
    clip_classification_logits,
    build_openclip_bundle,
)
from .model import StudentModel
from .loss import total_loss

TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a drawing of a {}.",
]


class PromptKD:
    """
    PromptKD — Prompt-based Knowledge Distillation for few-shot CLIP.

    Teacher: convnext_base_w (laion_aesthetic_s13b_b82k) — frozen.
    Student: ViT-B-32 (laion2b_s34b_b79k) — frozen except visual prompt + projector.
    Distillation data source: few-shot training split only.
    Projector maps student dim (512) → teacher dim (640) for KD loss.
    Classification uses raw student features (not projected).
    """

    method_name = "promptkd"

    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, train_loader, val_loader, classnames, model_bundle):
        """
        model_bundle = teacher (convnext_base_w).
        Student is loaded separately inside StudentModel.
        """
        device = model_bundle.device

        # Freeze teacher completely
        for param in model_bundle.model.parameters():
            param.requires_grad = False
        model_bundle.model.eval()

        # Precompute teacher text features once
        teacher_text_features = build_zero_shot_classifier(
            model_bundle, classnames, TEMPLATES
        )  # [num_classes, teacher_dim]

        # Build student text features using student tokenizer
        student_bundle = build_openclip_bundle(
            model_name=self.cfg.student_model_name,
            pretrained=self.cfg.student_pretrained,
            device=device,
            precision=self.cfg.precision,
        )
        student_text_features = build_zero_shot_classifier(
            student_bundle, classnames, TEMPLATES
        )  # [num_classes, student_dim]

        # Build student model
        student = StudentModel(cfg=self.cfg).to(device)

        print(f"Teacher text features: {teacher_text_features.shape}")
        print(f"Student text features: {student_text_features.shape}")

        # Only prompt and projector are trained
        optimizer = torch.optim.AdamW([
            {"params": student.visual_prompt.parameters()},
            {"params": student.projector.parameters()},
        ], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Training loop
        for epoch in range(self.cfg.epochs):
            student.train()
            total_loss_sum = 0
            batches        = 0

            for images, labels in self._iter_loader(train_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Teacher soft targets — no gradients
                with torch.no_grad():
                    t_feats  = model_bundle.model.encode_image(images)
                    t_feats  = F.normalize(t_feats, dim=-1)
                    t_logits = clip_classification_logits(
                        model_bundle, t_feats, teacher_text_features
                    )

                # Student forward
                # raw feats → classification loss (CE)
                # projected feats → KD loss against teacher
                s_feats, s_proj = student(images)

                # CE loss uses student text features + raw student feats
                s_logits_ce = self._logits(
                    s_feats, student_text_features,
                    student_bundle.model.logit_scale
                )

                # KD loss uses projected student feats vs teacher logits
                s_logits_kd = clip_classification_logits(
                    model_bundle, s_proj, teacher_text_features
                )

                loss, loss_ce, loss_kd = total_loss(
                    s_logits_ce, t_logits, labels,
                    temperature=self.cfg.temperature,
                    lambda_kd=self.cfg.lambda_kd,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss_sum += loss.item()
                batches        += 1

            avg = total_loss_sum / batches
            print(f"Epoch {epoch+1}/{self.cfg.epochs} | Avg Loss: {avg:.4f}")

        self._student              = student
        self._student_text_features = student_text_features
        self._student_bundle       = student_bundle
        self._model_bundle         = model_bundle

        return MethodArtifact(
            method_name=self.method_name,
            metadata={
                "teacher_model":      model_bundle.model_name,
                "teacher_pretrained": model_bundle.pretrained,
                "student_model":      self.cfg.student_model_name,
                "student_pretrained": self.cfg.student_pretrained,
                "distillation_data":  "few_shot_train_split_only",
                "epochs":             self.cfg.epochs,
                "temperature":        self.cfg.temperature,
                "lambda_kd":          self.cfg.lambda_kd,
            }
        )

    def evaluate(self, artifact, test_loader, classnames, model_bundle):
        """Evaluate using raw student features — not projected."""
        device  = model_bundle.device
        student = self._student
        student.eval()

        correct = 0
        total   = 0

        with torch.no_grad():
            for images, labels in self._iter_loader(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                # Raw features for classification
                feats, _ = student(images)
                logits   = self._logits(
                    feats,
                    self._student_text_features,
                    self._student_bundle.model.logit_scale
                )
                preds    = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        return {"test/top1_accuracy": accuracy}

    def _logits(self, image_features, text_features, logit_scale):
        """Compute classification logits from image and text features."""
        return logit_scale.exp() * image_features @ text_features.T

    def _iter_loader(self, loader):
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                yield batch[0], batch[1]
            else:
                yield batch["image"], batch["label"]
