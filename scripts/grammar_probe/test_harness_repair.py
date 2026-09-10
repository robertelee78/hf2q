#!/usr/bin/env python3
"""Offline harness regression suite; mock HTTP only, isolated temporary outputs.

Run: python3 -B scripts/grammar_probe/test_harness_repair.py
"""
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

if __name__ == "__main__":
    root = str(Path(__file__).resolve().parent)
    suite = unittest.TestSuite()
    for name in ("test_measurement_integrity.py", "test_judgment_pipeline.py"):
        suite.addTests(unittest.defaultTestLoader.discover(root, pattern=name))
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(not outcome.wasSuccessful())
