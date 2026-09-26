"""CPU-only checks for the standalone NFW-011 Colab replay notebook."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
NB_PATH = (
    ROOT
    / "neuralFirewallV2/experiments/NFW-11_authorization_provenance"
    / "NFW_011_Authorization_Provenance_Replay.ipynb"
)
NB = json.loads(NB_PATH.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"] if cell["cell_type"] == "code"}


class NFW011NotebookTest(unittest.TestCase):
    def test_notebook_cells_parse_hash_matches_and_broker_is_synced(self):
        code_payload = []
        for cell in NB["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source)
            self.assertEqual(cell["outputs"], [])
            self.assertIsNone(cell["execution_count"])
            code_payload.append((cell["id"], source))

        config = CELLS["configuration"]
        recorded = config.split("NOTEBOOK_CODE_SHA256 = '")[1].split("'")[0]
        actual = hashlib.sha256(
            json.dumps(
                [(cell_id, source.replace(recorded, "PENDING")) for cell_id, source in code_payload],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.assertEqual(recorded, actual)

        module_source = (
            ROOT / "neuralFirewallV2/src/policy/capability_broker.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(CELLS["broker"])
        broker_assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "BROKER_SOURCE" for target in node.targets)
        )
        self.assertEqual(ast.literal_eval(broker_assignment.value), module_source)

    def test_colab_notebook_replay_runs_cpu_only_on_frozen_results(self):
        input_dir = (
            ROOT
            / "neuralFirewallV2/experiments/NFW-10_protocol_corrected_injection/nfw-10-results"
        )
        self.assertTrue((input_dir / "manifest.json").exists())
        drive_stub = types.ModuleType("google.colab.drive")
        drive_stub.mount = lambda _: None
        google_stub = types.ModuleType("google")
        colab_stub = types.ModuleType("google.colab")
        colab_stub.drive = drive_stub
        google_stub.colab = colab_stub

        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            archive_path = temp / "nfw-10-results.zip"
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in input_dir.rglob("*"):
                    if path.is_file():
                        archive_name = (
                            "nfw-10-results/"
                            + path.relative_to(input_dir).as_posix()
                        )
                        archive.write(path, archive_name)
                # Ignore Finder metadata without mistaking it for the run tree.
                archive.writestr(
                    "__MACOSX/nfw-10-results/._manifest.json", b"not a JSON manifest"
                )
            ns = {"__name__": "__main__"}
            replacements = {
                "Path('/content/drive/MyDrive/NFW-010/nfw-10-results.zip')": f"Path({str(archive_path)!r})",
                "Path('/content/drive/MyDrive/NFW-011/nfw011_authorization_provenance_001')": f"Path({str(temp / 'drive_output')!r})",
                "Path('/content/nfw010_input')": f"Path({str(temp / 'extracted_input')!r})",
            }
            with patch.dict(
                sys.modules,
                {"google": google_stub, "google.colab": colab_stub, "google.colab.drive": drive_stub},
            ), contextlib.redirect_stdout(io.StringIO()):
                for cell_id in (
                    "configuration",
                    "input_validation",
                    "broker",
                    "replay",
                    "save_results",
                ):
                    source = CELLS[cell_id]
                    for old, new in replacements.items():
                        source = source.replace(old, new)
                    exec(compile(source, cell_id, "exec"), ns)

            summary = ns["SUMMARY"]
            self.assertEqual(summary["scope_only"]["injected"]["wrong_content_effects"], 7)
            self.assertEqual(summary["source_bound"]["injected"]["wrong_content_effects"], 0)
            self.assertEqual(summary["oracle_bound"]["injected"]["wrong_content_effects"], 0)
            self.assertTrue((temp / "drive_output/REPORT.md").exists())
            output = json.loads((temp / "drive_output/evaluation.json").read_text())
            self.assertEqual(len(output["rows"]), 216)

    def test_notebook_is_model_free(self):
        source = "\n".join(CELLS.values()).lower()
        self.assertNotIn("automodelforcausallm", source)
        self.assertNotIn("from transformers", source)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn("requests.get", source)


if __name__ == "__main__":
    unittest.main()
