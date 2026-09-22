#!/usr/bin/env python
"""Offline, minimal Coder--Tester molecular-agent demonstration.

This deliberately small example mirrors MolViBench's two-agent idea without an
LLM endpoint or package installation: a Coder proposes a simple function and a
Tester executes it on held-out molecules. It is useful for understanding the
agent loop before running the full API-backed ``inference_ac.py`` experiment.

Run: python mini_agent_demo.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MolecularTask:
    name: str
    instruction: str
    cases: tuple[tuple[str, float], ...]


class CoderAgent:
    """Produces a compact, dependency-free implementation for the task."""

    def propose(self, task: MolecularTask) -> str:
        if task.name != "molecular_weight":
            raise ValueError(f"Unsupported demo task: {task.name}")
        return """import re

ATOM_MASS = {"C": 12.011, "O": 15.999, "N": 14.007, "H": 1.008}
IMPLICIT_HYDROGENS = {"C": 4, "O": 2, "N": 3}

def level_function(smiles: str) -> float:
    atoms = re.findall(r"[CON]", smiles)
    if not atoms:
        raise ValueError(f\"Unsupported simple SMILES: {smiles}\")
    heavy_mass = sum(ATOM_MASS[atom] for atom in atoms)
    # This demo handles unbranched, single-bond C/O/N structures only.
    bonds = max(len(atoms) - 1, 0)
    hydrogens = sum(IMPLICIT_HYDROGENS[atom] for atom in atoms) - 2 * bonds
    return round(heavy_mass + hydrogens * ATOM_MASS["H"], 3)
"""


class TesterAgent:
    """Runs the proposal in a subprocess and reports an auditable verdict."""

    def verify(self, code: str, task: MolecularTask) -> list[str]:
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "proposal.py"
            module.write_text(code, encoding="utf-8")
            results: list[str] = []
            for smiles, expected in task.cases:
                runner = (
                    "import importlib.util; "
                    f"s=importlib.util.spec_from_file_location('p',{str(module)!r}); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                    f"print(m.level_function({smiles!r}))"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", runner],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                actual = completed.stdout.strip()
                passed = completed.returncode == 0 and actual == str(expected)
                status = "PASS" if passed else "FAIL"
                results.append(f"{status}: {smiles} -> {actual or completed.stderr.strip()} (expected {expected})")
            return results


def main() -> None:
    task = MolecularTask(
        name="molecular_weight",
        instruction="Return a molecule's molecular weight from a SMILES string.",
        cases=(("CCO", 46.069), ("O", 18.015)),
    )
    code = CoderAgent().propose(task)
    results = TesterAgent().verify(code, task)
    print(f"Task: {task.instruction}")
    print("Roles: Coder proposes code -> Tester validates held-out cases")
    print(*results, sep="\n")
    if any(line.startswith("FAIL") for line in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
