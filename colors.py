"""Terminal colors, so the different parts of the output are easy to tell apart.

Uses ANSI escape codes: invisible character sequences that terminals interpret as "switch to
green" or "reset". Because they're invisible but still count as characters, always pad text
for table columns *before* coloring it, or the columns won't line up.

Colors are on only when printing to a real terminal, so output piped to a file stays plain.
Set NO_COLOR=1 to turn them off, or FORCE_COLOR=1 to force them on (e.g. in the PyCharm console).
"""

import os
import sys

_force = os.environ.get("FORCE_COLOR")
if _force is not None:
    ENABLED = _force != "0"
else:
    ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ


def _style(code: str):
    def apply(text) -> str:
        return f"\033[{code}m{text}\033[0m" if ENABLED else str(text)

    return apply


bold = _style("1")
dim = _style("2")
red = _style("31")
green = _style("32")
yellow = _style("33")
blue = _style("34")
magenta = _style("35")
cyan = _style("36")


# What each color means, used the same way everywhere:
def header(text) -> str:  # section titles: "=== Answer ==="
    return bold(cyan(text))


def tool(text) -> str:  # searches and tool calls: "🔎 search_docs(...)"
    return blue(text)


def status(text) -> str:  # housekeeping: "Index up to date", token counts, file paths
    return dim(text)


def score(text) -> str:  # numbers that measure something: similarity scores, $ costs
    return yellow(text)


def good(text) -> str:
    return green(text)


def bad(text) -> str:
    return red(text)
