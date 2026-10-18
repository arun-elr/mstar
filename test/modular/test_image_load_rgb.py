"""The base image loader hands every model three channels, whatever the file holds."""
import io

import numpy as np
import pytest
import torch
from PIL import Image

from mstar.model.base import Model

RGB = (np.arange(48 * 64 * 3) % 251).astype(np.uint8).reshape(48, 64, 3)


def _save(path, mode, fmt):
    Image.fromarray(RGB).convert(mode).save(path, format=fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return str(path)


CASES = [("RGBA", "PNG", "rgba.png"), ("CMYK", "JPEG", "cmyk.jpg"), ("LA", "PNG", "gray_alpha.png"),
         ("L", "PNG", "gray.png"), ("P", "PNG", "palette.png"), ("RGB", "PNG", "rgb.png")]


@pytest.mark.parametrize(("mode", "fmt", "name"), CASES)
def test_image_loads_as_rgb(tmp_path, mode, fmt, name):
    path = _save(tmp_path / name, mode, fmt)
    out = Model.load_image(None, path, "cpu").data
    with Image.open(path) as image:
        expected = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0
    assert out.shape == (3, 48, 64)
    torch.testing.assert_close(out, expected, atol=1 / 255, rtol=0)


@pytest.mark.parametrize(("mode", "fmt", "name"), CASES)
def test_qwen_image_processor_accepts_it(tmp_path, mode, fmt, name):
    """Qwen3-Omni's image processor refused 4- and 2-channel images ("Unable
    to infer channel dimension format", a 400) and 1-channel ones (its
    3-channel mean, a 400 or a 500 depending on the transformers version)."""
    transformers = pytest.importorskip("transformers")
    out = Model.load_image(None, _save(tmp_path / name, mode, fmt), "cpu").data
    # Qwen3OmniModel.process_prompt's conversion: uint8, HWC for 1 or 3 channels
    img = (out * 255.0).clamp(0, 255).to(torch.uint8)
    if img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)
    processed = transformers.Qwen2VLImageProcessor()(images=[img.contiguous().numpy()], return_tensors="pt")
    assert processed["pixel_values"].shape[0] > 0


@pytest.mark.parametrize("fmt", ["PNG", "GIF"])  # GIF goes through Pillow, PNG through torchvision
def test_corrupt_image_is_a_client_error_without_the_path(tmp_path, fmt):
    buf = io.BytesIO()
    Image.fromarray(RGB).save(buf, format=fmt)
    path = tmp_path / f"secret-name.{fmt.lower()}"
    path.write_bytes(buf.getvalue()[: len(buf.getvalue()) // 2])  # the header survives, the pixels don't
    with pytest.raises(ValueError, match="corrupt or truncated") as info:
        Model.load_image(None, str(path), "cpu")
    assert "secret-name" not in str(info.value)
