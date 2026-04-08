from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .model import SmallUNet


class SegmentationPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str, image_size: int, base_channels: int = 48, threshold: float = 0.5) -> None:
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = SmallUNet(base_channels=base_channels).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.image_size = image_size
        self.threshold = threshold
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.inference_mode()
    def predict_mask(self, image: Image.Image) -> Image.Image:
        original_size = image.size
        resized = TF.resize(image.convert("RGB"), [self.image_size, self.image_size], interpolation=Image.Resampling.BILINEAR)
        tensor = self.normalize(TF.to_tensor(resized)).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        binary = (probs > self.threshold).astype(np.uint8) * 255
        mask = Image.fromarray(binary, mode="L")
        return mask.resize(original_size, resample=Image.Resampling.NEAREST)
