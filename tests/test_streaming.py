import json
import unittest

from filament.llm import AnthropicClient, OllamaClient, OpenAICompatibleClient, _anthropic_messages
from filament.stream import parse_tool_arguments


class FakeStream:
    def __init__(self, lines, content_type="text/event-stream"):
        self.lines = [line.encode("utf-8") if isinstance(line, str) else line for line in lines]
        self.headers = {"Content-Type": content_type}

    def __iter__(self):
        return iter(self.lines)


def sse(payload):
    return "data: " + json.dumps(payload)


class TestOpenAIStream(unittest.TestCase):
    def test_text_and_fragmented_tool_call_are_assembled(self):
        client = OpenAICompatibleClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [
            sse({"choices": [{"delta": {"content": "He"}}]}),
            sse({"choices": [{"delta": {"content": "llo"}}]}),
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": '{"pa'}}]}}]}),
            sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'th": "x.py"}'}}]}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]",
        ]
        deltas = []
        result = client._read_stream(FakeStream(lines), deltas.append, None)
        self.assertEqual(result.content, "Hello")
        self.assertEqual(deltas, ["He", "llo"])
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "x.py"})
        self.assertEqual(result.tool_calls[0].id, "call_1")

    def test_malformed_chunks_are_skipped(self):
        client = OpenAICompatibleClient(model="m", base_url="http://localhost:1", stream=True)
        result = client._read_stream(
            FakeStream(["data: not-json", sse({"choices": [{"delta": {"content": "ok"}}]})]),
            None,
            None,
        )
        self.assertEqual(result.content, "ok")

    def test_malformed_chunk_shapes_are_skipped(self):
        client = OpenAICompatibleClient(model="m", base_url="http://localhost:1", stream=True)
        result = client._read_stream(
            FakeStream(
                [
                    sse({"choices": [{"delta": "nope"}]}),
                    sse({"usage": "nope", "choices": [{"delta": {"content": "ok"}}]}),
                    sse({"choices": [{"delta": {"content": "!"}}]}),
                ]
            ),
            None,
            None,
        )
        self.assertEqual(result.content, "ok!")

    def test_cancel_stops_reading(self):
        client = OpenAICompatibleClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [sse({"choices": [{"delta": {"content": "one"}}]}), sse({"choices": [{"delta": {"content": "two"}}]})]
        seen = []
        result = client._read_stream(FakeStream(lines), seen.append, lambda: True)
        self.assertEqual(result.content, "")
        self.assertEqual(seen, [])


class TestAnthropicStream(unittest.TestCase):
    def test_text_and_tool_use_blocks(self):
        client = AnthropicClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [
            sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}}),
            sse({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "tu_1", "name": "grep"}}),
            sse({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"pattern": "x"}'}}),
            sse({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
        ]
        result = client._read_stream(FakeStream(lines), None, None)
        self.assertEqual(result.content, "Hi")
        self.assertEqual(result.tool_calls[0].name, "grep")
        self.assertEqual(result.tool_calls[0].arguments, {"pattern": "x"})
        self.assertEqual(result.finish_reason, "tool_use")
        self.assertEqual(result.usage["completion_tokens"], 7)

    def test_malformed_event_shapes_are_skipped(self):
        client = AnthropicClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [
            sse({"type": "content_block_start", "index": "bad", "content_block": "nope"}),
            sse({"type": "content_block_delta", "index": "bad", "delta": "nope"}),
            sse({"type": "message_delta", "delta": "nope", "usage": "nope"}),
            sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}),
            sse({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ]
        result = client._read_stream(FakeStream(lines), None, None)
        self.assertEqual(result.content, "ok")

    def test_messages_are_converted_and_tool_results_grouped(self):
        system, conversation = _anthropic_messages(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "t1", "name": "read_file", "arguments": {"path": "a.py"}}],
                },
                {"role": "tool", "tool_call_id": "t1", "content": "data"},
                {"role": "tool", "tool_call_id": "t2", "content": "data2"},
                {"role": "user", "content": "next"},
            ]
        )
        self.assertEqual(system, "sys")
        self.assertEqual(conversation[1]["content"][0]["type"], "tool_use")
        grouped = conversation[2]
        self.assertEqual(grouped["role"], "user")
        # A user message after the tool results must be merged into the same
        # turn: strict backends reject consecutive user roles.
        self.assertEqual(
            [block["type"] for block in grouped["content"]],
            ["tool_result", "tool_result", "text"],
        )
        self.assertEqual(grouped["content"][-1]["text"], "next")
        self.assertEqual(len(conversation), 3)


class TestOpenAIMessageConversion(unittest.TestCase):
    def test_assistant_tool_calls_are_serialized(self):
        from filament.llm import _openai_message

        message = _openai_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}
                ],
            }
        )
        self.assertIsNone(message["content"])
        call = message["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "read_file")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"path": "a.py"})

    def test_tool_result_maps_to_tool_role(self):
        from filament.llm import _openai_message

        message = _openai_message(
            {"role": "tool", "tool_call_id": "c1", "content": "data"}
        )
        self.assertEqual(message, {"role": "tool", "tool_call_id": "c1", "content": "data"})


class TestOllamaStream(unittest.TestCase):
    def test_ndjson_text_and_tool_calls(self):
        client = OllamaClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [
            json.dumps({"message": {"content": "Hi"}, "done": False}),
            json.dumps(
                {
                    "message": {
                        "content": "",
                        "tool_calls": [{"function": {"name": "list_dir", "arguments": {"path": "."}}}],
                    },
                    "done": False,
                }
            ),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2}),
        ]
        result = client._read_stream(FakeStream(lines, "application/x-ndjson"), None, None)
        self.assertEqual(result.content, "Hi")
        self.assertEqual(result.tool_calls[0].arguments, {"path": "."})
        self.assertEqual(result.usage["prompt_tokens"], 3)
        self.assertEqual(result.finish_reason, "stop")

    def test_malformed_message_shapes_are_skipped(self):
        client = OllamaClient(model="m", base_url="http://localhost:1", stream=True)
        lines = [
            json.dumps({"message": "nope", "done": False}),
            json.dumps({"message": {"content": "ok"}, "done": False}),
            json.dumps({"message": None, "done": True, "done_reason": "stop"}),
        ]
        result = client._read_stream(FakeStream(lines, "application/x-ndjson"), None, None)
        self.assertEqual(result.content, "ok")
        self.assertEqual(result.finish_reason, "stop")


class TestToolArguments(unittest.TestCase):
    def test_accepts_objects_and_json_strings(self):
        self.assertEqual(parse_tool_arguments({"a": 1}), ({"a": 1}, ""))
        self.assertEqual(parse_tool_arguments('{"a": 1}'), ({"a": 1}, ""))

    def test_rejects_invalid_payloads(self):
        args, error = parse_tool_arguments("[1, 2]")
        self.assertEqual(args, {})
        self.assertIn("list", error)
        args, error = parse_tool_arguments("{bad")
        self.assertIn("invalid JSON", error)
        args, error = parse_tool_arguments("[]")
        self.assertIn("JSON object", error)


if __name__ == "__main__":
    unittest.main()
