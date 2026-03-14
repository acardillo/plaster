import io

import open_clip
import torch
from PIL import Image, UnidentifiedImageError
from fastapi import UploadFile

_model = None
_preprocess = None


class ModelLoadError(Exception):
    """Raised when the CLIP model fails to load."""


def _load_model():
    global _model, _preprocess
    if _model is None:
        try:
            _model, _, _preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            _model.eval()
        except Exception as e:
            raise ModelLoadError("Model failed to load") from e
    return _model, _preprocess


async def get_embedding(image: UploadFile) -> list[float]:
    model, preprocess = _load_model()
    contents = await image.read()
    if not contents:
        raise ValueError("Empty file")
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("Invalid or unsupported image") from e
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].tolist()