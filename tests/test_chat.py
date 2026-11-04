import unittest

from filament.chat import MAX_PLAN_TASKS, _extract_json, _sanitize_target


class TestChatPlanning(unittest.TestCase):
    def test_extract_json_from_markdown_fence(self):
        raw = "```json\n{\"summary\": \"s\", \"tasks\": [\"a\"]}\n```"
        data = _extract_json(raw)
        self.assertEqual(data["summary"], "s")
        self.assertEqual(data["tasks"], ["a"])

    def test_extract_json_stops_at_first_object(self):
        raw = '{"summary": "s", "tasks": ["a"]} extra trailing junk {not json}'
        data = _extract_json(raw)
        self.assertEqual(data["summary"], "s")

    def test_sanitizes_unsafe_targets(self):
        self.assertEqual(_sanitize_target("../secret.py"), "output.py")
        self.assertEqual(_sanitize_target("/abs/path.py"), "output.py")
        self.assertEqual(_sanitize_target("C:drive.py"), "output.py")
        self.assertEqual(_sanitize_target(""), "output.py")
        self.assertEqual(_sanitize_target(None), "output.py")
        self.assertEqual(_sanitize_target("nested/out.py"), "nested/out.py")

    def test_sanitizes_null_bytes(self):
        self.assertEqual(_sanitize_target("safe\x00evil.py"), "output.py")

    def test_max_plan_tasks_cap(self):
        self.assertEqual(MAX_PLAN_TASKS, 6)

    def test_fallback_when_no_json(self):
        # _extract_json raises before fallback; ChatOrchestrator handles it.
        with self.assertRaises(ValueError):
            _extract_json("not json at all")


if __name__ == "__main__":
    unittest.main()
