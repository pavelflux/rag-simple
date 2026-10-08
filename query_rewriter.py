"""Query decomposition: a small, cheap model splits a multi-part question into sub-questions."""

import json

from generator import create_client

REWRITER_MODEL = "claude-haiku-4-5"  # a simple task, so the cheapest, fastest model is enough

SYSTEM_PROMPT = """You prepare a user's question for a semantic search over company documents
(employee handbook, company overview, product FAQ, information security policy).

- If the question is about one topic, return it unchanged, word for word, as the only item.
- If it asks about several separate things, split it into self-contained sub-questions in the
  user's own words, e.g. "What's the hotel limit in Paris?" and "How fast do robots move near people?"
- Don't add words, topics or details the user didn't mention, and don't rephrase into keywords.
- Don't split one topic into pieces: "hotel and meal allowance" is one topic (travel expenses).
- Return at most 4 items."""

# Structured outputs: the reply is guaranteed to be {"queries": [...]}, so parsing can't fail.
SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
    "additionalProperties": False,
}


class QueryRewriter:
    """Turns one user question into a list of short search queries, one per topic."""

    def __init__(self, model: str = REWRITER_MODEL):
        self.client = create_client()
        self.model = model

    def split(self, question: str) -> list[str]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,  # the reply is a short JSON list
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next(block.text for block in response.content if block.type == "text")
        queries = json.loads(text)["queries"]
        return queries or [question]  # never end up with nothing to search for
