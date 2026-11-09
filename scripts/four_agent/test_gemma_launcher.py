"""Gemma launcher behavior against current and pre-budget hf2q binaries."""

import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "serve_gemma4_opencode.sh"


class GemmaLauncher(unittest.TestCase):
    def run_launcher(self, supports_budget: bool, decode_dense: str | None = None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            model.write_bytes(b"fixture")
            binary = root / "hf2q"
            binary.write_text(
                "#!/bin/bash\n"
                "set -eu\n"
                "if [[ \"$*\" == \"serve --help\" ]]; then\n"
                "  [[ \"${FAKE_SUPPORTS_BUDGET:-0}\" == 1 ]] && echo --lcp-resume-budget || echo serve\n"
                "  exit 0\n"
                "fi\n"
                "printf '%s\\n' \"$@\" > \"$CAPTURE\"\n"
                "printf '%s\\n' \"${HF2Q_GEMMA_FULL_INPUT_CHECKPOINT-}\" > \"$CAPTURE.env\"\n"
                "printf '%s\\n' \"${HF2Q_GEMMA_DECODE_DENSE-}\" > \"$CAPTURE.dense\"\n"
            )
            binary.chmod(0o755)
            capture = root / "argv"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            env = {
                **os.environ,
                "CAPTURE": str(capture),
                "FAKE_SUPPORTS_BUDGET": "1" if supports_budget else "0",
                "FULL_INPUT_CHECKPOINT": "0",
                "HF2Q_BIN": str(binary),
                "KV_CACHE_BUDGET_BYTES": "1024",
                "MAX_SLOTS": "1",
                "MMPROJ": str(root / "absent-mmproj.gguf"),
                "MODEL": str(model),
                "PORT": str(port),
            }
            if decode_dense is not None:
                env["DECODE_DENSE"] = decode_dense
            result = subprocess.run([str(LAUNCHER)], env=env, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            return (
                capture.read_text().splitlines(),
                (root / "argv.env").read_text().strip(),
                (root / "argv.dense").read_text().strip(),
                result.stderr,
            )

    def test_versioned_budget_flag_and_environment_forwarding(self):
        current, forwarded, dense, current_stderr = self.run_launcher(True)
        self.assertEqual(forwarded, "0")
        self.assertEqual(dense, "")
        position = current.index("--lcp-resume-budget")
        self.assertEqual(current[position + 1], "8GiB")
        self.assertNotIn("predates --lcp-resume-budget", current_stderr)

        previous, forwarded, dense, previous_stderr = self.run_launcher(False)
        self.assertEqual(forwarded, "0")
        self.assertEqual(dense, "")
        self.assertNotIn("--lcp-resume-budget", previous)
        self.assertIn("predates --lcp-resume-budget", previous_stderr)
        self.assertIn("--kv-cache-budget", previous)
        self.assertIn("--default-repetition-penalty", previous)

    def test_decode_dense_is_forwarded_only_when_explicit(self):
        _, _, dense, _ = self.run_launcher(True, decode_dense="row_exact")
        self.assertEqual(dense, "row_exact")


if __name__ == "__main__":
    unittest.main()
