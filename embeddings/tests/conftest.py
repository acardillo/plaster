import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image


@pytest.fixture
def mock_clip():
    """Swap the real CLIP model for lightweight fakes."""
    fake_model = MagicMock()
    fake_model.encode_image.return_value = torch.randn(1, 512)

    fake_preprocess = MagicMock(side_effect=lambda img: torch.randn(3, 224, 224))

    with patch("pipeline._load_model", return_value=(fake_model, fake_preprocess)):
        yield fake_model, fake_preprocess


def make_image_bytes(mode: str = "RGB", size: tuple[int, int] = (64, 64)) -> bytes:
    """Generate a minimal in-memory PNG for testing."""
    color = "red" if mode != "L" else 128
    img = Image.new(mode, size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()
