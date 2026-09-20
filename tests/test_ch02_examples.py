import subprocess
import sys
from pathlib import Path

import pytest


EXAMPLES = sorted(Path("ch02/examples").glob("[0-9]*.py"))


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_chapter_2_example_help_does_not_download_a_model(example):
    completed = subprocess.run(
        [sys.executable, str(example), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
