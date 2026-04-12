import torch
import torch.nn as nn
from common.models.openclip import encode_image_features, encode_text_features


class DPCModel(nn.Module):
    def __init__(self, bundle, classnames):
        super().__init__()
        self.bundle = bundle
        self.classnames = classnames
        self.device = bundle.device

        # Trainable prompt features
        with torch.no_grad():
            texts = [f"a photo of a {c}" for c in classnames]
            init_features = encode_text_features(bundle, texts)
            self.trainable_prompt = nn.Parameter(init_features.clone())

        # Reference text features (frozen)
        with torch.no_grad():
            texts = [f"a photo of a {c}" for c in classnames]
            ref_features = encode_text_features(bundle, texts)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)

        self.register_buffer("reference_text_features", ref_features)

    def forward(self, images):
        image_features = encode_image_features(self.bundle, images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Trainable logits
        text_features = self.trainable_prompt
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T

        # Reference logits
        reference_logits = image_features @ self.reference_text_features.T

        return logits, reference_logits