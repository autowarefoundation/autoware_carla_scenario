"""The version bump label check accepts exactly one bump label (#2055).

``release.yml`` picks the largest bump when a PR carries several, so a PR
labelled both ``bump patch`` and ``bump major`` must fail the check rather
than release a major version silently.  The check is the inline
``actions/github-script`` script of ``pr-auto-check.yml``; these tests run it
under Node with a fake ``github``, ``context`` and ``core``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_WORKFLOW = (
    Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pr-auto-check.yml"
)

#: github-script runs the script as the body of an async function.
_HARNESS = """
const labels = JSON.parse(process.argv[1]);
const failures = [];
const github = { rest: { pulls: { get: async () => ({
  data: { labels: labels.map(name => ({ name })) },
}) } } };
const context = { repo: { owner: 'o', repo: 'r' }, issue: { number: 1 } };
const core = { setFailed: message => failures.push(message) };
console.log = () => {};
(async (github, context, core) => {
%s
})(github, context, core).then(
  () => process.stdout.write(JSON.stringify(failures)),
  error => { process.stderr.write(String(error)); process.exit(2); },
);
"""


@pytest.fixture(scope="module")
def script() -> str:
    workflow: dict[str, Any] = yaml.safe_load(_WORKFLOW.read_text())
    (step,) = workflow["jobs"]["check-version-bump-label"]["steps"]
    assert step["uses"].startswith("actions/github-script@")
    return str(step["with"]["script"])


def _failures(script: str, labels: list[str]) -> list[str]:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "-e", _HARNESS % script, json.dumps(labels)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    failures: list[str] = json.loads(result.stdout)
    return failures


@pytest.mark.parametrize("label", ["bump patch", "bump minor", "bump major"])
def test_one_bump_label_passes(script: str, label: str) -> None:
    assert _failures(script, ["documentation", label]) == []


def test_no_bump_label_fails(script: str) -> None:
    (failure,) = _failures(script, ["documentation"])
    assert "exactly one" in failure
    assert "bump patch" in failure


def test_two_bump_labels_fail(script: str) -> None:
    (failure,) = _failures(script, ["bump patch", "bump major"])
    assert "2 version bump labels" in failure
    assert "bump patch, bump major" in failure


def test_all_bump_labels_fail(script: str) -> None:
    (failure,) = _failures(script, ["bump major", "bump minor", "bump patch"])
    assert "3 version bump labels" in failure
