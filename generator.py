"""Step 5: ask Claude to answer a question using the retrieved chunks."""

import os
from dataclasses import dataclass

import anthropic

from vector_store import SearchResult

MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You answer questions about a company's internal documents.

The relevant document excerpts are given inside <context>. Follow these rules:
- Use only information from the context for company rules and facts, even if you think you know the answer.
- If applying a rule needs a general fact the context doesn't contain (for example a flight duration), you may use it, but say clearly that it's an assumption and not from the documents.
- Cite the source file name in square brackets after each fact, e.g. [employee_handbook.md].
- If the context does not contain the answer, reply: "I don't know based on the provided documents."
- If the context answers only part of the question, answer that part and say which part you could not find.
- Keep answers short and direct."""


@dataclass
class Answer:
    text: str
    input_tokens: int
    output_tokens: int


def create_client() -> anthropic.Anthropic:
    """An Anthropic client; reads ANTHROPIC_API_KEY automatically."""
    # Keys that aren't tied to a workspace must say which workspace to use on every request.
    workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    return anthropic.Anthropic(default_headers=headers)


def format_documents(results: list[SearchResult]) -> str:
    """Wrap each chunk in a <document> tag carrying its source file and section."""
    documents = []
    for result in results:
        chunk = result.chunk
        documents.append(
            f'<document source="{chunk.source}" section="{chunk.heading}">\n'
            f"{chunk.text}\n"
            f"</document>"
        )
    return "\n\n".join(documents)


class Generator:
    """Sends the question plus retrieved chunks to Claude and returns its answer."""

    def __init__(self, model: str = MODEL):
        self.client = create_client()
        self.model = model

    def build_prompt(self, question: str, results: list[SearchResult]) -> str:
        """Put the chunks inside <context>, then the question last."""
        return f"<context>\n{format_documents(results)}\n</context>\n\nQuestion: {question}"

    def answer(self, question: str, results: list[SearchResult]) -> Answer:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": self.build_prompt(question, results)}],
            # "low" effort = less thinking before answering. Plenty for reading a few chunks.
            output_config={"effort": "low"},
            # If a safety check declines the request, retry on a fallback model automatically.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

        if response.stop_reason == "refusal":
            text = "(Claude declined to answer this request.)"
        else:
            # The response is a list of blocks (thinking, text, ...); we only want the text.
            text = "".join(block.text for block in response.content if block.type == "text")

        return Answer(text, response.usage.input_tokens, response.usage.output_tokens)
