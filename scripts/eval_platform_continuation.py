"""Replay the platform entry point using public projections and a mocked model reply."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness_image"))
import contest_submission as submission


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


class LocalRound:
    max_output_tokens = 64

    def __init__(self, key, index, checkpoint, train_path, validation_path, reply="SUBMIT"):
        self.dataset_key = key
        self.round_index = index
        self.task = {"task": "classification" if key == "A" else "regression"}
        self.parent = SimpleNamespace(
            artifact_id="parent", descriptor={"checkpoint_paths": ["checkpoint.npz"]},
            path=lambda _: checkpoint,
        )
        self.data = SimpleNamespace(
            training_rows=lambda: read_rows(train_path),
            validation_rows=lambda: read_rows(validation_path),
        )
        self.reply = reply
        self.calls = 0
        self.candidate = None

    def model(self, messages, **options):
        self.calls += 1
        return SimpleNamespace(text=self.reply, call_id="mock-local")

    def history(self):
        return []

    def submit_candidate(self, directory, **declaration):
        self.candidate = directory / declaration["checkpoint_paths"][0]
        return {"artifact_id": "candidate"}

    def finish(self, **record):
        assert self.calls == 1
        assert record["selected_artifact_id"] == ("candidate" if self.candidate else "parent")
        self.record = record["configuration"]


def c_reference_test(parent, candidate):
    output = {}
    base = ROOT / "data/external/cmapss"
    for subset in ("FD001", "FD002", "FD003", "FD004"):
        table = np.loadtxt(base / f"test_{subset}.txt")
        target = np.loadtxt(base / f"RUL_{subset}.txt")
        last = table[np.r_[table[:-1, 0] != table[1:, 0], True]]
        sample = [dict(settings=row[2:5].tolist(), sensors=row[5:26].tolist()) for row in last]
        predictions = lambda model: np.asarray([model.predict_one(x)["value"] for x in sample])
        scores = {}
        for name, model in (("parent", parent), ("candidate", candidate)):
            err = predictions(model) - target
            scores[name] = {"rmse": float(np.sqrt(np.mean(err ** 2))),
                            "mae": float(np.abs(err).mean())}
        output[subset] = scores
    return output


def main():
    report = {}
    for key in "ABC":
        folder = next((ROOT / "data/official").glob(f"cpu-{key.lower()}-*"))
        initial = folder / "checkpoint.npz"
        checkpoint = initial
        rounds = []
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"CONTEST_WORK": tmp}):
            for index in (1, 2):
                rnd = LocalRound(key, index, checkpoint, folder / "public/train.jsonl",
                                 folder / "public/validation.jsonl")
                submission.train(rnd)
                rounds.append(rnd.record)
                if rnd.candidate:
                    model = submission.load_model(rnd.candidate)
                    row = next(read_rows(folder / "public/validation.jsonl"))
                    with patch.dict(os.environ, {"CONTEST_ARTIFACT_PATH": str(rnd.candidate)}):
                        actual = submission.predict(SimpleNamespace(inputs=row["inputs"]), tmp)
                        assert actual == model.predict_one(row["inputs"])
                    checkpoint = rnd.candidate
                    if key == "C" and index == 1:
                        report["C_reference_test"] = c_reference_test(submission.load_model(initial), model)
                print(key, index, "submitted", rnd.record["submitted"],
                      rnd.record.get("parent_metrics", rnd.record.get("public_parent")),
                      rnd.record.get("selected_candidate", rnd.record.get("public_candidate")), flush=True)
            if key in "BC":
                assert rounds[0]["tested_candidates"] > 0
                assert rounds[0]["submitted"], rounds[0]
            report[key] = rounds
    path = ROOT / "results/official/platform_continuation"
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
