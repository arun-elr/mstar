"""Keep the two Dockerfiles' pins consistent with each other and with pyproject.toml."""

import re
import tomllib
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "mstar/integrations/dynamo/docker"
DOCKERFILES = [DOCKER_DIR / "Dockerfile", DOCKER_DIR / "Dockerfile.dynamo-base"]
PIN_ARGS = ("TORCH_INDEX", "TORCH_VERSION", "TORCHVISION_VERSION", "TORCHAUDIO_VERSION",
            "TORCHCODEC_VERSION", "FLASH_ATTN_WHEEL")


def version(text):
    return tuple(int(p) for p in re.findall(r"\d+", text))


def build_args(path):
    return dict(re.findall(r"^ARG (\w+)=(\S+)$", path.read_text(), re.MULTILINE))


def mstar_install(path):
    """The `uv pip install ... -e ".[all...]"` RUN step, continuation lines joined."""
    text = path.read_text().replace("\\\n", " ")
    match = re.search(r'^RUN uv pip install [^\n]*-e "\.\[all[^\n]*$', text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{path.name}: no -e .[all] install step")
    return match.group(0)


class DockerPinTests(unittest.TestCase):
    def test_both_dockerfiles_pin_the_same_torch_stack(self):
        first, second = (build_args(p) for p in DOCKERFILES)
        for name in PIN_ARGS:
            self.assertIn(name, first)
            self.assertEqual(first[name], second.get(name), name)

    def test_torch_stack_matches_the_pyproject_floor_and_itself(self):
        args = build_args(DOCKERFILES[0])
        deps = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
        floor = next(d for d in deps if d.startswith("torch>="))
        self.assertGreaterEqual(version(args["TORCH_VERSION"]), version(floor.split(">=")[1].split(",")[0]))
        torch_minor = version(args["TORCH_VERSION"])[1]
        # torchcodec 0.N pairs with torch 2.N
        self.assertEqual(version(args["TORCHCODEC_VERSION"])[1], torch_minor)
        cuda_major = re.search(r"cu(\d+?)\d$", args["TORCH_INDEX"]).group(1)
        self.assertIn(f"+cu{cuda_major}torch2.{torch_minor}cxx11", args["FLASH_ATTN_WHEEL"])

    def test_runtime_setuptools_meets_the_build_floor(self):
        # torch pulls setuptools from the torch index, whose copy is older than the build floor
        requires = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["build-system"]["requires"]
        build_floor = version(next(r for r in requires if r.startswith("setuptools>=")).split(">=")[1])
        for path in DOCKERFILES:
            pin = re.search(r'"setuptools>=([\d.]+)"', mstar_install(path))
            self.assertIsNotNone(pin, f"{path.name}: no setuptools floor in the .[all] install")
            self.assertGreaterEqual(version(pin.group(1)), build_floor, path.name)


if __name__ == "__main__":
    unittest.main()
