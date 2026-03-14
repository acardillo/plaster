import io

import open_clip
import torch
from PIL import Image
from fastapi import UploadFile

_model = None
_preprocess = None


def _load_model():
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _model.eval()
    return _model, _preprocess


async def get_embedding(image: UploadFile) -> list[float]:
    model, preprocess = _load_model()
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model.encode_image(tensor)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].tolist()