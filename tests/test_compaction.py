import unittest

from filament.compaction import (
    SUMMARY_MARKER,
    compact_messages,
    estimate_tokens,
    prune_tool_results,
)


def long_conversation(turns=12):
    messages = [{"role": "system", "content": "system prompt"}]
    for index in range(turns):
        messages.append({"role": "user", "content": f"request {index} " + "x" * 400})
        messages.append(
            {
                "role": "assistant",
                "content": f"answer {index} " + "y" * 400,
                "tool_calls": [
                    {
                        "id": f"c{index}",
                        "name": "read_file",
                        "arguments": {"path": "app.py"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"c{index}",
                "name": "read_file",
                "content": "z" * 2000,
            }
        )
    return messages


class TestCompaction(unittest.TestCase):
    def test_compaction_summarizes_old_turns_and_keeps_recent(self):
        messages = long_conversation()
        before = estimate_tokens(messages)
        compacted, changed, stats = compact_messages(
            None,
            messages,
            max_tokens=1500,
            chars_per_token=4,
            keep_recent=6,
            threshold=0.5,
            use_llm=False,
        )
        self.assertTrue(changed)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertIn(SUMMARY_MARKER, compacted[1]["content"])
        self.assertLess(stats["after_tokens"], before)
        self.assertEqual(stats["used_llm"], False)
        self.assertGreater(stats["summarized_messages"], 0)
        # The recent window must start at a user turn, never a dangling tool result.
        self.assertEqual(compacted[2]["role"], "user")
        self.assertEqual(compacted[-1]["content"], messages[-1]["content"])

    def test_no_compaction_below_threshold(self):
        messages = long_conversation(turns=2)
        compacted, changed, stats = compact_messages(
            None, messages, max_tokens=100000, keep_recent=4, threshold=0.75
        )
        self.assertFalse(changed)
        self.assertEqual(compacted, messages)
        self.assertEqual(stats["pruned_results"], 0)

    def test_prune_replaces_large_old_tool_results(self):
        messages = long_conversation(turns=3)
        pruned, count = prune_tool_results(messages, keep_recent=3, limit_chars=100)
        self.assertGreater(count, 0)
        self.assertIn("omitted during compaction", pruned[3]["content"])
        # The last three messages stay intact.
        self.assertEqual(pruned[-1]["content"], messages[-1]["content"])

    def test_compaction_never_orphans_a_tool_result(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "a", "name": "read_file", "arguments": {"path": "x"}}],
            },
            {"role": "tool", "tool_call_id": "a", "content": "data"},
            {"role": "user", "content": "next"},
            {"role": "assistant", "content": "ok"},
        ]
        compacted, changed, _ = compact_messages(
            None, messages, max_tokens=20, chars_per_token=4, keep_recent=2, threshold=0.1
        )
        # keep_recent=2 lands inside the tool result; the cut must advance past it.
        self.assertTrue(changed)
        self.assertNotEqual(compacted[-2]["role"], "tool")


if __name__ == "__main__":
    unittest.main()
