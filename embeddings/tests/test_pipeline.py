import io

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from tests.conftest import make_image_bytes
from pipeline import get_embedding


def _upload(data: bytes, filename: str = "poster.png") -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(data))


class TestGetEmbedding:
    async def test_returns_list_of_floats(self, mock_clip):
        result = await get_embedding(_upload(make_image_bytes()))
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    async def test_dimension_is_512(self, mock_clip):
        result = await get_embedding(_upload(make_image_bytes()))
        assert len(result) == 512

    async def test_l2_normalized(self, mock_clip):
        result = await get_embedding(_upload(make_image_bytes()))
        assert float(np.linalg.norm(result)) == pytest.approx(1.0, abs=1e-5)

    async def test_accepts_rgba(self, mock_clip):
        result = await get_embedding(_upload(make_image_bytes(mode="RGBA")))
        assert len(result) == 512

    async def test_accepts_grayscale(self, mock_clip):
        result = await get_embedding(_upload(make_image_bytes(mode="L")))
        assert len(result) == 512

    async def test_converts_to_rgb(self, mock_clip):
        _, fake_preprocess = mock_clip
        await get_embedding(_upload(make_image_bytes(mode="RGBA")))
        img_arg = fake_preprocess.call_args[0][0]
        assert isinstance(img_arg, Image.Image)
        assert img_arg.mode == "RGB"

    async def test_calls_encode_image(self, mock_clip):
        fake_model, _ = mock_clip
        await get_embedding(_upload(make_image_bytes()))
        fake_model.encode_image.assert_called_once()

    async def test_empty_file_raises(self, mock_clip):
        with pytest.raises(ValueError, match="Empty file"):
            await get_embedding(_upload(b""))

    async def test_invalid_image_raises(self, mock_clip):
        with pytest.raises(ValueError, match="Invalid or unsupported image"):
            await get_embedding(_upload(b"not an image"))
