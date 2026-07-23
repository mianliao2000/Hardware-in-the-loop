from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from gui.server import LLM_COMPLETION_MARKER, _call_llm_chat


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._data = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._data


class LlmChatTest(unittest.TestCase):
    def test_missing_marker_is_continued_even_when_provider_says_stop(self) -> None:
        responses = [
            _FakeHttpResponse(
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": "The system is ready to"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
            _FakeHttpResponse(
                {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {"content": f"run the next iteration. {LLM_COMPLETION_MARKER}"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
        ]
        environment = {
            "OPENROUTER_API_KEY": "test-key",
            "WEB_SEARCH_ENABLED": "false",
            "LLM_MAX_TOKENS": "10000",
            "LLM_MAX_CONTINUATIONS": "2",
        }
        with patch.dict(os.environ, environment, clear=False), patch(
            "gui.server.urllib_request.urlopen", side_effect=responses
        ) as urlopen:
            reply, model, completion = _call_llm_chat(
                [{"role": "user", "content": "Explain the run state."}],
                {},
                "gemini-3.5-flash",
            )

        self.assertEqual(reply, "The system is ready to\nrun the next iteration.")
        self.assertEqual(model, "test-model")
        self.assertTrue(completion["complete"])
        self.assertEqual(completion["continuations"], 1)
        self.assertEqual(completion["completion_protocol"], "sentinel-v1")
        self.assertEqual(urlopen.call_count, 2)
        continuation_body = json.loads(urlopen.call_args_list[1].args[0].data.decode("utf-8"))
        self.assertEqual(continuation_body["max_tokens"], 10000)
        self.assertEqual(continuation_body["messages"][-2]["role"], "assistant")
        self.assertIn("Continue exactly", continuation_body["messages"][-1]["content"])

    def test_completed_reply_strips_marker_from_visible_text(self) -> None:
        response = _FakeHttpResponse(
            {
                "model": "test-model",
                "choices": [
                    {
                        "message": {"content": f"Complete answer. {LLM_COMPLETION_MARKER}"},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        environment = {
            "OPENROUTER_API_KEY": "test-key",
            "WEB_SEARCH_ENABLED": "false",
            "LLM_MAX_TOKENS": "10000",
            "LLM_MAX_CONTINUATIONS": "4",
        }
        with patch.dict(os.environ, environment, clear=False), patch(
            "gui.server.urllib_request.urlopen", return_value=response
        ) as urlopen:
            reply, _model, completion = _call_llm_chat(
                [{"role": "user", "content": "Answer completely."}],
                {},
                "gemini-3.5-flash",
            )

        self.assertEqual(reply, "Complete answer.")
        self.assertEqual(completion["continuations"], 0)
        self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
