from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "sap_audit", ROOT / "workflows/two_stage_navigation/audit_s0010_sap_geometry.py"
)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SapGeometryAuditTests(unittest.TestCase):
    def test_unique_projected_pixels_do_not_count_duplicate_vertices(self) -> None:
        rows = np.asarray([10, 10, 10, 11])
        columns = np.asarray([20, 20, 20, 20])
        self.assertEqual(AUDIT.unique_pixel_count(rows, columns), 2)

    def test_asset_audit_does_not_promote_whole_vertebra_to_sap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stl_root = root / "selected_dataset_stl/s0010"
            usd_root = root / "selected_dataset_body_from_urdf/s0010/combined_wrapwrap"
            stl_root.mkdir(parents=True)
            usd_root.mkdir(parents=True)
            for level in AUDIT.LEVELS:
                (stl_root / f"vertebrae_{level}.stl").write_bytes(level.encode())
            (stl_root / "combined_label_map.nii.gz").write_bytes(b"labels")
            (stl_root / "s0010.urdf").write_text(
                '<robot><link name="base_link"><mesh filename="combined_wrapwrap.stl"/></link></robot>'
            )
            (usd_root / "combined_wrapwrap.usd").write_text("def Xform combined_wrapwrap {}")
            result = AUDIT.audit_asset_semantics(root)
            self.assertEqual(result["process_or_facet_named_files"], [])
            self.assertFalse(result["urdf_has_separate_process_or_facet_name"])
            self.assertFalse(result["usd_has_separate_process_or_facet_name"])
            self.assertTrue(result["urdf_uses_combined_body_mesh"])

    def test_geometry_loader_keeps_proposed_and_uncertain_masks_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.npz"
            np.savez_compressed(
                path,
                vertices_mesh_index=np.zeros((3, 3), dtype=np.float32),
                faces=np.asarray([[0, 1, 2]], dtype=np.int32),
                original_selected_vertex_mask=np.asarray([True, True, False]),
                proposed_revised_vertex_mask=np.asarray([False, False, False]),
                boundary_uncertainty_vertex_mask=np.asarray([True, True, False]),
                removed_vertex_mask=np.zeros(3, dtype=bool),
                added_vertex_mask=np.zeros(3, dtype=bool),
                source_mesh_sha256=np.asarray("0" * 64),
            )
            result = AUDIT.load_geometry(path)
            self.assertEqual(np.count_nonzero(result["proposed"]), 0)
            self.assertEqual(np.count_nonzero(result["uncertain"]), 2)
            self.assertFalse(np.shares_memory(result["proposed"], result["uncertain"]))


if __name__ == "__main__":
    unittest.main()
