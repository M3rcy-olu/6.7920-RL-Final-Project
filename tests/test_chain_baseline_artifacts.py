"""Check the committed study artifacts independently of training."""
import json
from pathlib import Path

import pytest

from mini_envs.chain_worlds import CHAIN_WORLDS, COST_WORLDS
from q_learning import Policy, evaluate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "q_learning/policies/chain_results.json"


def test_study_world_coverage():
    rows = json.loads(RESULTS.read_text())
    expected = {key for key, value in
                {**CHAIN_WORLDS, **COST_WORLDS}.items()
                if value["family"] != "matter"}
    assert len(rows) == len(expected) == 14
    assert {row["env_id"] for row in rows} == expected
    assert {row["world"] for row in rows if not row["passed"]} == {"Cost-Graded-8"}


@pytest.mark.parametrize("world", [
    "Base", "Delay-L1", "Delay-L8", "Delay-L16", "Doors-3", "Doors-4",
    "Rooms-2", "Rooms-6", "Rooms-8", "Size-7", "Hard", "Cost-Graded",
    "Cost-Graded-Rev", "Cost-Graded-8",
])
def test_saved_policy_quality(world):
    rows = json.loads(RESULTS.read_text())
    row = next(row for row in rows if row["world"] == world)
    if world == "Cost-Graded-8":
        assert not row["passed"]
        assert not (ROOT / row["file"]).exists()
        return
    assert row["passed"]
    policy = Policy.load(ROOT / row["file"])
    assert policy.env_id == row["env_id"]
    assert policy.max_steps == row["max_steps"] == 50 * row["optimal_steps"]
    metrics = evaluate(policy)
    assert metrics["success_rate"] == 1
    assert metrics["mean_return"] == pytest.approx(1)
    assert metrics["unseen_episodes"] == 0
    assert metrics["mean_steps"] <= 1.1 * row["optimal_steps"]
    for key, value in metrics.items():
        assert value == row["attempts"][-1][key]
