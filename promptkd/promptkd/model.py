
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class VisualPrompt(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        prompt = torch.zeros(3, image_size, image_size)
        nn.init.normal_(prompt, std=0.01)
        self.prompt = nn.Parameter(prompt)

    def forward(self, x):
        return x + self.prompt


class StudentModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = cfg.device

        # Load student backbone — separate from teacher
        student_clip, _, _ = open_clip.create_model_and_transforms(
            cfg.student_model_name,
            pretrained=cfg.student_pretrained,
            device=device,
        )
        self.encoder     = student_clip
        self.logit_scale = student_clip.logit_scale

        # Freeze student encoder completely
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # Get student embed dim dynamically
        with torch.no_grad():
            dummy   = torch.zeros(1, 3, cfg.image_size, cfg.image_size).to(device)
            out     = self.encoder.encode_image(dummy)
            student_dim = out.shape[-1]

        # Get teacher embed dim — needed for projector
        teacher_clip, _, _ = open_clip.create_model_and_transforms(
            cfg.teacher_model_name,
            pretrained=cfg.teacher_pretrained,
            device=device,
        )
        with torch.no_grad():
            t_out       = teacher_clip.encode_image(dummy)
            teacher_dim = t_out.shape[-1]
        del teacher_clip  # free memory immediately

        print(f"Student dim: {student_dim} | Teacher dim: {teacher_dim}")

        # Projector: maps student dim → teacher dim for KD loss
        self.projector     = nn.Linear(student_dim, teacher_dim, bias=False)
        self.visual_prompt = VisualPrompt(cfg.image_size)

        self.student_dim = student_dim
        self.teacher_dim = teacher_dim

    def forward(self, x):
        # Apply visual prompt
        x = self.visual_prompt(x)

        # Encode through frozen student encoder
        feats     = self.encoder.encode_image(x)
        feats     = F.normalize(feats, dim=-1)

        # Project into teacher feature space for KD loss
        projected = self.projector(feats)
        projected = F.normalize(projected, dim=-1)

        return feats, projected  # raw feats for classification, projected for KD
