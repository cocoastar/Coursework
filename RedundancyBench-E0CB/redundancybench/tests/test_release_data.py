"""Check the published dataset scope and the minimal trajectory schema."""

import json
from pathlib import Path
import unittest

from redundancybench.data_loader import load_dataset


DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
MESSAGE_FIELDS = {"role", "content", "tool_calls", "id", "requestor", "error"}
TASK_FIELDS = {"id", "description", "user_scenario", "evaluation_criteria"}


class ReleaseDataTests(unittest.TestCase):
    def test_scope_and_anonymous_runtime_metadata(self):
        dataset = load_dataset(DATA_ROOT)
        self.assertEqual(len(dataset.trajectories), 1920)
        self.assertEqual(len(dataset.cases), 3613)
        for trajectory in dataset.trajectories:
            for message in trajectory.messages:
                self.assertLessEqual(message.keys(), MESSAGE_FIELDS)
        for path in DATA_ROOT.glob("*/*/final_traces.json"):
            traces = json.loads(path.read_text(encoding="utf-8"))
            for task in traces["tasks"]:
                self.assertLessEqual(task.keys(), TASK_FIELDS)
            for simulation in traces["simulations"]:
                self.assertNotIn("provider_session_id", simulation)
                self.assertNotIn("timestamp", simulation)


if __name__ == "__main__":
    unittest.main()
