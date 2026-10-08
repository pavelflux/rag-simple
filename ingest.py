"""Step 2: load documents from ./docs and split them into overlapping chunks."""

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


def count_tokens(text: str) -> int:
    """Rough token estimate: English text averages about 1.3 tokens per word."""
    return round(len(text.split()) * 1.3)


@dataclass
class Document:
    source: str  # file name, used for citations
    text: str  # the full text of the file


@dataclass
class Chunk:
    text: str  # the content that gets embedded and shown to Claude
    source: str  # file name, used for citations
    heading: str  # section the chunk came from ("" if the file has no headings)
    index: int  # position of the chunk within its file


class DocumentLoader:
    """Reads every supported file in a folder into Document objects."""

    SUPPORTED = {".md", ".txt", ".pdf"}

    def __init__(self, docs_dir: Path = Path("docs")):
        self.docs_dir = docs_dir

    def load(self) -> list[Document]:
        return [
            Document(source=path.name, text=self._read(path))
            for path in sorted(self.docs_dir.iterdir())
            if path.suffix in self.SUPPORTED
        ]

    def _read(self, path: Path) -> str:
        if path.suffix == ".pdf":
            pages = [page.extract_text() for page in PdfReader(path).pages]
            return "\n\n".join(pages)
        return path.read_text(encoding="utf-8")


class Chunker:
    """Splits documents into sections (at headings), then packs each section into chunks.

    document → sections → paragraphs → chunks (≤ chunk_size tokens, with overlap)
    """

    def __init__(self, chunk_size: int = 200, overlap: int = 40):
        self.chunk_size = chunk_size  # target maximum tokens per chunk
        self.overlap = overlap  # max tokens carried over from the end of the previous chunk

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks = []
        for doc in documents:
            index = 0
            for heading, body in self._split_sections(doc.text):
                for text in self._pack(self._split_paragraphs(body)):
                    chunks.append(Chunk(text, doc.source, heading, index))
                    index += 1
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Split at headings into (heading, body) pairs.

        Two kinds of headings start a new section:
        - markdown headings ("## Paid Time Off"). Nested ones become a path like
          "Handbook > Paid Time Off > Sick Leave".
        - FAQ-style question lines: a short line that ends with "?" and follows a blank line,
          as in our product_faq.txt. Each question and its answer become their own section,
          so one chunk covers one Q&A instead of mixing several topics.
        Files with neither (our .pdf) come back as one section with heading "".
        """
        sections = []
        heading_stack = []  # the current markdown heading at each level
        question = ""  # the current FAQ question, if any
        lines = []

        def flush():
            body = "\n".join(lines).strip()
            if body:  # skip headings that have no text directly under them
                sections.append((" > ".join(heading_stack + ([question] if question else [])), body))
            lines.clear()

        previous_blank = True  # the start of the text counts as following a blank line
        for line in text.splitlines():
            match = re.match(r"^(#{1,6})\s+(.*)", line)
            if match:
                flush()
                level = len(match.group(1))
                heading_stack[:] = heading_stack[: level - 1] + [match.group(2).strip()]
                question = ""
            elif self._is_question(line, previous_blank):
                flush()
                question = line.strip()
                # Unlike markdown headings, keep the question in the chunk text too: its wording
                # is usually the closest match to how users ask, so it should be embedded.
                lines.append(line)
            else:
                lines.append(line)
            previous_blank = not line.strip()
        flush()
        return sections

    @staticmethod
    def _is_question(line: str, previous_blank: bool) -> bool:
        """An FAQ question line: starts a paragraph, ends with "?", and is short."""
        stripped = line.strip()
        return previous_blank and stripped.endswith("?") and len(stripped) <= 120

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split on blank lines. Also strips the padding PDF extraction adds to each line."""
        paragraphs = re.split(r"\n\s*\n", text)
        cleaned = ["\n".join(line.strip() for line in p.splitlines()).strip() for p in paragraphs]
        return [p for p in cleaned if p]

    def _split_sentences(self, text: str) -> list[str]:
        return re.split(r"(?<=[.!?])\s+", text)

    def _overlap_tail(self, text: str) -> str:
        """Return the last whole sentences of text, up to self.overlap tokens."""
        tail = []
        for sentence in reversed(self._split_sentences(text)):
            if count_tokens(" ".join([sentence] + tail)) > self.overlap:
                break
            tail.insert(0, sentence)
        return " ".join(tail)

    def _pack(self, paragraphs: list[str]) -> list[str]:
        """Greedily add paragraphs to a chunk until the next one would exceed chunk_size.

        Each new chunk starts with the last sentence(s) of the previous one (the overlap),
        so a fact that sits on a chunk border is not cut off from its context.
        """
        # A single paragraph that is too big on its own gets broken into sentences.
        pieces = []
        for para in paragraphs:
            if count_tokens(para) > self.chunk_size:
                pieces.extend(self._split_sentences(para))
            else:
                pieces.append(para)

        chunks = []
        current = []
        for piece in pieces:
            if current and count_tokens("\n\n".join(current + [piece])) > self.chunk_size:
                finished = "\n\n".join(current)
                chunks.append(finished)
                overlap = self._overlap_tail(finished)
                fits = overlap and count_tokens(overlap) + count_tokens(piece) <= self.chunk_size
                current = [overlap] if fits else []
            current.append(piece)
        if current:
            chunks.append("\n\n".join(current))
        return chunks


# ---------- Demo ----------


def print_chunk(chunk: Chunk) -> None:
    heading = chunk.heading or "(no heading)"
    print(f"── {chunk.source} #{chunk.index} · {heading} · ~{count_tokens(chunk.text)} tokens")
    print(chunk.text)
    print()


if __name__ == "__main__":
    show_all = "--all" in sys.argv

    documents = DocumentLoader().load()
    chunks = Chunker(chunk_size=200, overlap=40).chunk(documents)

    print(f"Loaded {len(documents)} documents → {len(chunks)} chunks\n")
    for doc in documents:
        sizes = [count_tokens(c.text) for c in chunks if c.source == doc.source]
        print(f"  {doc.source:<24} {len(sizes):>3} chunks, {min(sizes)}–{max(sizes)} tokens each")
    print()

    # By default show the first 2 chunks of each file; --all shows everything.
    for doc in documents:
        doc_chunks = [c for c in chunks if c.source == doc.source]
        for chunk in doc_chunks if show_all else doc_chunks[:2]:
            print_chunk(chunk)
