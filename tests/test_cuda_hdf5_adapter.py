# SPDX-License-Identifier: GPL-2.0-or-later
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from parosol_cuda import STRAIN_COMPONENTS, main, solve


class AdapterTest(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.directory = Path(workspace.name)
        self.source = self.directory / "cpu.h5"
        self.output = self.directory / "gpu.h5"
        with h5py.File(self.source, "w") as file:
            file.attrs["fixture"] = "linear ParOSol"
            group = file.create_group("Image_Data")
            group["Image"] = np.full((1, 1, 1), 350.0, dtype=np.float32)
            group["Voxelsize"] = [0.3]
            group["Poisons_ratio"] = [0.25]
            group["Fixed_Displacement_Coordinates"] = np.zeros((1, 4), dtype=np.uint16)
            group["Fixed_Displacement_Values"] = np.zeros(1, dtype=np.float32)
            file["Solution/old"] = [123]

    def write_native_result(self, command, **kwargs):
        self.assertTrue(kwargs["check"])
        self.assertEqual(command[:3], ["native-solver", "--tol", "1e-8"])
        with h5py.File(command[-1], "r+") as native:
            self.assertNotIn("Solution", native)
            self.assertNotIn("Mesh", native)
            self.assertEqual(native["Image_Data/Poisons_ratio"][...].item(), 0.25)
            moduli = native["Image_Data/Image"][...]
            self.assertEqual(moduli.item(), np.float32(0.35))
            native["Mesh/Coordinates"] = 0.3 * np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
            )
            native["Mesh/Elements"] = np.arange(8).reshape(1, 8)
            native["Mesh/Emoduli"] = [[1000 * float(moduli.item())]]
            native["Solution/disp"] = np.zeros((8, 3))
            native["Solution/force"] = np.zeros((8, 3))
            for name in ("SED", "EFF", "VonMises"):
                native[f"Solution/{name}"] = [[0.0]]
            for index, name in enumerate(STRAIN_COMPONENTS, 1):
                native[f"Solution/{name}"] = [[0.001 * index]]

    def test_export_preserves_input_and_matches_cpu_schema(self):
        original = self.source.read_bytes()
        with patch("parosol_cuda.subprocess.run", side_effect=self.write_native_result):
            solve(self.source, self.output, "native-solver", ["--tol", "1e-8"])
        self.assertEqual(self.source.read_bytes(), original)
        with h5py.File(self.output) as result:
            self.assertEqual(result.attrs["fixture"], "linear ParOSol")
            self.assertNotIn("old", result["Solution"])
            self.assertEqual(result["Image_Data/Image"][0, 0, 0], 350)
            np.testing.assert_array_equal(result["Mesh/Elements"][0], np.arange(1, 9))
            self.assertEqual(result["Mesh/Material IDs"].dtype, np.int32)
            self.assertEqual(result["Mesh/Material IDs"][0, 0], 350)
            np.testing.assert_allclose(result["Solution/Element strain"][0], np.arange(1, 7) * 0.001)
            # At E=350 MPa and nu=0.25, both Lame constants are 140 MPa.
            expected = np.array([1.12, 1.40, 1.68, 0.56, 0.70, 0.84])
            np.testing.assert_allclose(result["Solution/Element stress"][0], expected, rtol=1e-7)

    def test_relative_external_image_is_preserved_in_another_directory(self):
        with h5py.File(self.directory / "moduli.h5", "w") as materials, h5py.File(self.source, "r+") as model:
            materials["E"] = model["Image_Data/Image"][...]
            del model["Image_Data/Image"]
            model["Image_Data/Image"] = h5py.ExternalLink("moduli.h5", "/E")
        output = self.directory / "results" / "gpu.h5"
        output.parent.mkdir()
        with patch("parosol_cuda.subprocess.run", side_effect=self.write_native_result):
            solve(self.source, output, "native-solver", ["--tol", "1e-8"])
        with h5py.File(output, "r") as result:
            self.assertIsInstance(result.get("Image_Data/Image", getlink=True), h5py.HardLink)
            self.assertEqual(result["Image_Data/Image"][0, 0, 0], 350)

    def test_rejects_invalid_poisson_ratio_before_launch(self):
        for ratio in (-0.01, 0.5, np.nan):
            with self.subTest(ratio=ratio):
                with h5py.File(self.source, "r+") as file:
                    file["Image_Data/Poisons_ratio"][0] = ratio
                with patch("parosol_cuda.subprocess.run") as native:
                    with self.assertRaisesRegex(ValueError, r"Poisons_ratio must be in \[0, 0\.5\)"):
                        solve(self.source, self.output, "native-solver", [])
                    native.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_native_mode_passes_the_original_path_without_conversion(self):
        with patch("parosol_cuda.subprocess.run") as native, \
                patch("parosol_cuda.prepare_native_input") as prepare, \
                patch("parosol_cuda.tempfile.TemporaryDirectory") as staging:
            status = main(["--format", "native", "--solver", "native-solver",
                           "--tol", "1e-8", str(self.source)])
        self.assertEqual(status, 0)
        self.assertEqual(native.call_args.args[0], ["native-solver", "--tol", "1e-08", str(self.source)])
        self.assertTrue(native.call_args.kwargs["check"])
        prepare.assert_not_called()
        staging.assert_not_called()

    def test_native_mode_rejects_an_extra_output_path(self):
        with patch("parosol_cuda.subprocess.run") as native, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as failure:
                main(["--format", "native", str(self.source), str(self.output)])
        self.assertEqual(failure.exception.code, 2)
        native.assert_not_called()

    def test_failed_or_incomplete_solver_output_is_unpublished(self):
        original = self.source.read_bytes()
        with patch("parosol_cuda.subprocess.run", side_effect=subprocess.CalledProcessError(2, "solver")):
            with self.assertRaises(subprocess.CalledProcessError):
                solve(self.source, self.output, "native-solver", [])
        with patch("parosol_cuda.subprocess.run"):
            with self.assertRaises(KeyError):
                solve(self.source, self.output, "native-solver", [])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.directory.iterdir()), [self.source])

    def test_refuses_existing_output(self):
        with patch("parosol_cuda.subprocess.run") as native:
            with self.assertRaises(FileExistsError):
                solve(self.source, self.source, "native-solver", [])
            native.assert_not_called()


if __name__ == "__main__":
    unittest.main()
