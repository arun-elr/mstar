"""Check the CPU test selection without requiring PyTorch or Docker."""

import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTOR = REPO_ROOT / "scripts/ci/select_cpu_tests.py"
SELECTOR_NS = runpy.run_path(str(SELECTOR))
SELECT_TESTS = SELECTOR_NS["select_tests"]
CPU_TEST_PATHS = SELECTOR_NS["CPU_TEST_PATHS"]


def manifest_lines(name):
    return [line.strip() for line in (REPO_ROOT / name).read_text().splitlines() if line.strip()]


class CpuSelectionTests(unittest.TestCase):
    def test_no_hand_kept_cpu_test_list(self):
        self.assertFalse(
            (REPO_ROOT / "test/cpu-core.txt").exists(),
            "test/cpu-core.txt is gone: CPU Core runs every module under test/modular. "
            "Don't list tests; make a module that needs a GPU, weights or an optional "
            "package skip itself (see CONTRIBUTING.md).",
        )

    def test_github_job_runs_the_same_paths(self):
        workflow = (REPO_ROOT / ".github/workflows/ci.yml").read_text().splitlines()
        commands = [line.strip() for line in workflow if not line.strip().startswith("#")]
        self.assertIn(" ".join(CPU_TEST_PATHS), commands)

    def test_real_manifest_excludes_only_documented_incompatible_modules(self):
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(set(excluded), {
            "test/modular/test_waypoint_components.py",
            "test/modular/test_waypoint_dit.py",
            "test/modular/test_flex_attention_resource.py",
            "test/modular/test_waypoint_benchmark_videos.py",
        })
        self.assertEqual(
            SELECT_TESTS(list(CPU_TEST_PATHS), excluded),
            list(CPU_TEST_PATHS) + [f"--ignore={path}" for path in excluded],
        )
        for path in CPU_TEST_PATHS:
            self.assertTrue((REPO_ROOT / path).exists(), path)
        for path in excluded:
            self.assertTrue((REPO_ROOT / path).is_file(), path)

    def test_exclusions_become_ignores_and_listed_files_are_dropped(self):
        self.assertEqual(
            SELECT_TESTS(["tests", "extra.py"], ["tests/arm.py", "extra.py"]),
            ["tests", "--ignore=tests/arm.py"],
        )

    def test_manifest_reader_ignores_blank_lines_and_surrounding_whitespace(self):
        read_manifest = SELECTOR_NS["read_manifest"]
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "tests.txt"
            manifest.write_text("\n first.py \n\nlast.py\n \n")
            self.assertEqual(read_manifest(manifest), ["first.py", "last.py"])

    def test_unknown_exclusion_fails(self):
        for excluded in ("unknown.py", "testsuite/a.py"):
            with self.subTest(excluded=excluded):
                with self.assertRaisesRegex(ValueError, f"absent from CPU test paths: {excluded}"):
                    SELECT_TESTS(["tests"], [excluded])

    def test_duplicate_exclusions_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate paths"):
            SELECT_TESTS(["tests"], ["tests/a.py", "tests/a.py"])

    def test_empty_selection_fails(self):
        with self.assertRaisesRegex(ValueError, "selection is empty"):
            SELECT_TESTS(["only.py"], ["only.py"])

    def test_cli_from_other_directory_separates_paths_and_explanation(self):
        with tempfile.TemporaryDirectory() as other_directory:
            result = subprocess.run(
                [sys.executable, str(SELECTOR)], cwd=other_directory,
                capture_output=True, text=True, check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        excluded = manifest_lines("test/cpu-mac-exclude.txt")
        self.assertEqual(result.stdout.splitlines(), SELECT_TESTS(list(CPU_TEST_PATHS), excluded))
        self.assertIn("requires AVX2", result.stderr)
        self.assertIn(f"Excluded {len(excluded)} modules", result.stderr)
        for path in excluded:
            self.assertIn(path, result.stderr)


if __name__ == "__main__":
    unittest.main()
