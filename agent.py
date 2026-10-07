"""Step 7: agentic RAG. Claude gets a search tool and decides what to search, and how often."""

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from generator import MODEL, create_client, format_documents
from vector_store import SearchResult

RAW_RESPONSES_FILE = "agent_responses.json"  # written by run(..., show_responses=True)

SYSTEM_PROMPT = """You answer questions about a company's internal documents.
You cannot see the documents directly. Use the search_docs tool to find relevant excerpts.

- Search before answering. Split multi-part or vague questions into several focused searches.
- If the results don't contain what you need, search again with different wording.
- Use only information from search results. Do not use outside knowledge.
- Cite the source file name in square brackets after each fact, e.g. [employee_handbook.md].
- If you can't find the answer after a few searches, reply: "I don't know based on the provided documents."
- Keep answers short and direct."""

# The tool definition is all Claude knows about the tool: its name, what it does, what input it takes.
SEARCH_TOOL = {
    "name": "search_docs",
    "description": (
        "Semantic search over the company's documents: employee handbook, company overview, "
        "product FAQ, and information security policy. Returns the most relevant excerpts, "
        "each with its source file and section. Use a short query about one topic."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for, e.g. 'parental leave length'",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,  # guarantees Claude's tool input matches the schema exactly
}

# Every tool Claude may use. Each one needs a matching entry in Agent.tool_handlers.
TOOLS = [SEARCH_TOOL]


@dataclass
class AgentAnswer:
    text: str
    tool_calls: list[str] = field(default_factory=list)  # every tool call Claude made, in order
    turns: int = 0  # number of API calls
    input_tokens: int = 0
    output_tokens: int = 0


class Agent:
    """Runs the agent loop: call Claude → run the tools it asks for → send results back → repeat."""

    def __init__(
        self,
        search: Callable[[str, int], list[SearchResult]],
        model: str = MODEL,
        k: int = 4,
        max_turns: int = 6,
    ):
        self.search = search  # the RAG.search method from rag.py
        self.client = create_client()
        self.model = model
        self.k = k
        self.max_turns = max_turns  # safety limit so a confused agent can't loop forever

        # Tool name (as in the tool definition) → method that runs it.
        # To add a tool: add its definition to TOOLS and its method here.
        self.tool_handlers = {
            "search_docs": self._search_docs,
        }

    def _run_tool(self, name: str, tool_input: dict, verbose: bool) -> tuple[str, bool]:
        """Run one tool call. Returns (content, is_error).

        Every tool_use must get a tool_result, even when something goes wrong, otherwise the
        next API call is rejected. So failures become error results that Claude can read.
        """
        handler = self.tool_handlers.get(name)
        if handler is None:
            return f"Unknown tool: {name}", True
        try:
            return handler(tool_input, verbose), False
        except Exception as error:
            return f"Tool {name} failed: {error}", True

    def _search_docs(self, tool_input: dict, verbose: bool) -> str:
        query = tool_input["query"]
        results = self.search(query, self.k)
        if verbose:
            sources = ", ".join(f"{r.chunk.source}#{r.chunk.index}" for r in results)
            print(f"  🔎 search_docs({query!r}) → {sources}")
        return format_documents(results)

    def run(self, question: str, verbose: bool = True, show_responses: bool = False) -> AgentAnswer:
        answer = AgentAnswer(text="")
        # The conversation so far. It grows every turn and is sent in full on every call.
        messages = [{"role": "user", "content": question}]
        raw_responses = []  # every API response as plain JSON, saved to a file at the end

        for _ in range(self.max_turns):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
                # Thinking is always on; "summarized" returns a readable summary of it instead of
                # an empty block. It only changes what we see, not how Claude thinks.
                thinking={"type": "adaptive", "display": "summarized" if show_responses else "omitted"},
                # "medium" leaves room to plan several searches; step 5 only had to read.
                output_config={"effort": "medium"},
                # If a safety check declines the request, retry on a fallback model automatically.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            answer.turns += 1
            answer.input_tokens += response.usage.input_tokens
            answer.output_tokens += response.usage.output_tokens
            if show_responses:
                self._print_response(response, answer.turns)
                raw_responses.append(response.to_dict())  # the SDK object as a plain dict

            # Keep Claude's whole reply (thinking, text, tool calls) in the conversation.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break  # Claude answered (or refused) instead of asking for another search

            # Claude may ask for several tool calls in one reply. Run them all and send
            # every result back together in a single user message.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                answer.tool_calls.append(f"{block.name}({block.input})")
                content, is_error = self._run_tool(block.name, block.input, verbose)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,  # links this result to the call that asked for it
                    "content": content,
                    "is_error": is_error,  # True tells Claude the call failed, so it can adapt
                })
            messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "refusal":
            answer.text = "(Claude declined to answer this request.)"
        elif response.stop_reason == "tool_use":
            answer.text = f"(Stopped after {self.max_turns} turns without a final answer.)"
        else:
            answer.text = "".join(block.text for block in response.content if block.type == "text")

        if show_responses:
            with open(RAW_RESPONSES_FILE, "w", encoding="utf-8") as f:
                json.dump(raw_responses, f, indent=2, ensure_ascii=False)
            print(f"\n  Raw responses saved to {RAW_RESPONSES_FILE}")
        return answer

    @staticmethod
    def _print_response(response, turn: int) -> None:
        """Show every content block of one Claude response, as Claude returned it."""
        usage = response.usage
        print(
            f"\n  ┌─ Claude response, call {turn} "
            f"(stop_reason: {response.stop_reason}, {usage.input_tokens} in / {usage.output_tokens} out)"
        )
        for block in response.content:
            if block.type == "thinking":
                print(f"  │ [thinking] {block.thinking.strip() or '(empty)'}")
            elif block.type == "text":
                print(f"  │ [text] {block.text.strip()}")
            elif block.type == "tool_use":
                print(f"  │ [tool_use] {block.name}({block.input})  id={block.id}")
            else:
                print(f"  │ [{block.type}]")
        print("  └─")
