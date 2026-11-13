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


def single_long_turn(steps=12, size=2000):
    """One user request followed by many tool steps; no later user boundary."""
    messages: list[dict] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "one long task " + "x" * 200},
    ]
    for index in range(steps):
        messages.append(
            {
                "role": "assistant",
                "content": f"step {index}",
                "tool_calls": [
                    {
                        "id": f"s{index}",
                        "name": "read_file",
                        "arguments": {"path": "a.py"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"s{index}",
                "name": "read_file",
                "content": "z" * size,
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

    def test_single_long_turn_is_compacted_under_budget(self):
        # One user request with many tool steps has no later user boundary; it
        # must still compact instead of growing past the budget forever.
        messages = single_long_turn()
        before = estimate_tokens(messages)
        compacted, changed, stats = compact_messages(
            None,
            messages,
            max_tokens=1500,
            chars_per_token=4,
            keep_recent=8,
            threshold=0.5,
            use_llm=False,
        )
        self.assertTrue(changed)
        self.assertIn(SUMMARY_MARKER, compacted[1]["content"])
        self.assertIn("one long task", compacted[1]["content"])
        self.assertLessEqual(stats["after_tokens"], 1500)
        self.assertLess(stats["after_tokens"], before)
        self.assertNotEqual(compacted[2]["role"], "tool")
        self.assert_paired(compacted)

    def test_mid_turn_drops_keep_tool_pairing(self):
        messages = single_long_turn(steps=8, size=4000)
        compacted, changed, _ = compact_messages(
            None,
            messages,
            max_tokens=600,
            chars_per_token=4,
            keep_recent=8,
            threshold=0.5,
            use_llm=False,
        )
        self.assertTrue(changed)
        self.assertNotEqual(compacted[0]["role"], "tool")
        self.assert_paired(compacted)

    def assert_paired(self, messages):
        pending = set()
        for message in messages:
            if message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    pending.add(call["id"])
            elif message.get("role") == "tool":
                self.assertIn(message.get("tool_call_id"), pending)
                pending.discard(message.get("tool_call_id"))
        self.assertEqual(pending, set())

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

    def test_wide_tool_batch_at_the_tail_is_still_summarized(self):
        # Regression: when the newest turn is one wide tool batch (spawn_agents,
        # a parallel read/grep fan-out) every message after the keep window is a
        # tool result.  Advancing past them used to hit the end of the
        # conversation, so the old code returned without summarizing and the
        # request stayed far over budget.
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "read the modules and report"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"b{index}", "name": "read_file", "arguments": {"path": f"m{index}.py"}}
                    for index in range(10)
                ],
            },
        ]
        for index in range(10):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"b{index}",
                    "name": "read_file",
                    "content": "z" * 4000,
                }
            )
        before = estimate_tokens(messages)
        compacted, changed, stats = compact_messages(
            None,
            messages,
            max_tokens=6000,
            chars_per_token=4,
            keep_recent=8,
            threshold=0.75,
            use_llm=False,
        )
        self.assertTrue(changed)
        self.assertGreater(stats["summarized_messages"], 0)
        self.assertIn(SUMMARY_MARKER, compacted[1]["content"])
        self.assertLess(stats["after_tokens"], before)
        self.assertLessEqual(stats["after_tokens"], 6000)
        # The batch survives with its pairing intact; only the fat outputs shrink.
        batch = next(message for message in compacted if message.get("tool_calls"))
        self.assertEqual(len(batch["tool_calls"]), 10)
        self.assert_paired(compacted)
        self.assertNotEqual(compacted[2]["role"], "tool")

    def test_short_history_over_budget_prunes_instead_of_giving_up(self):
        # Regression: keep_recent larger than the conversation leaves no turn to
        # summarize.  Compaction used to return the prompt unchanged and over
        # budget even though pruning can shrink it without breaking pairing.
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "inspect everything"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"s{index}", "name": "read_file", "arguments": {"path": f"f{index}.py"}}
                    for index in range(6)
                ],
            },
        ]
        for index in range(6):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"s{index}",
                    "name": "read_file",
                    "content": "z" * 3000,
                }
            )
        before = estimate_tokens(messages)
        compacted, changed, stats = compact_messages(
            None,
            messages,
            max_tokens=1000,
            chars_per_token=4,
            keep_recent=12,
            threshold=0.75,
            use_llm=False,
        )
        self.assertTrue(changed)
        self.assertGreater(stats["pruned_results"], 0)
        self.assertLess(stats["after_tokens"], before)
        self.assert_paired(compacted)


if __name__ == "__main__":
    unittest.main()
