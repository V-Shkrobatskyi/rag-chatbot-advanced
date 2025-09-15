import app.llm_utils as llm_utils
from typing import Any

# Simple mock object returned by create()
mock_response: Any = type(
    "Response",
    (),
    {
        "choices": [
            type(
                "Choice",
                (),
                {"message": type("Msg", (), {"content": "Mocked answer test"})()},
            )
        ]
    },
)()


def test_ask_llm(monkeypatch: Any) -> None:
    """Test that ask_llm returns the mocked response."""

    class MockChatCompletion:
        @staticmethod
        def create(**kwargs: Any) -> Any:
            # Return the pre-defined mock response
            return mock_response

    class MockOpenAIClient:
        # Provide a mock chat.completions interface
        chat: Any = type("Chat", (), {"completions": MockChatCompletion()})()

    # Replace the real openai_client with the mock
    monkeypatch.setattr(llm_utils, "openai_client", MockOpenAIClient())

    answer: str = llm_utils.ask_llm("What is AI?")
    assert isinstance(answer, str)
    assert "test" in answer
