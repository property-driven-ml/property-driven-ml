"""Regression test: the documented entrypoint must at least parse its arguments.

`python main.py --help` raised argparse.ArgumentError because --epsilon and
--delta were each defined twice. That made every invocation of the training
script fail before any work started.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_main_help_exits_cleanly():
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "conflicting option string" not in result.stderr
    assert "--epsilon" in result.stdout


def test_main_defines_each_flag_once():
    source = (ROOT / "main.py").read_text()
    for flag in ('"--epsilon"', '"--delta"'):
        assert source.count(flag) == 1, f"{flag} defined more than once"
