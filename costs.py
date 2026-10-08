"""Step 10: what does a run cost? Token prices, plus a tracker that records every API call."""

from dataclasses import dataclass, field

import colors

# $ per million tokens: (input, output). From Anthropic's pricing page; update when prices change.
# Includes the fallback models, in case a request is served by one of them.
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
}


def price_for(model: str) -> tuple[float, float]:
    # Longest name first, so "claude-sonnet-5-5" isn't mistaken for "claude-sonnet-5".
    for name in sorted(PRICES, key=len, reverse=True):
        if model.startswith(name):
            return PRICES[name]
    raise ValueError(f"No price for model {model!r}; add it to PRICES in costs.py")


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = price_for(model)
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class CallCost:
    """One API call: which model answered, why it stopped, and how many tokens it used."""

    model: str
    stop_reason: str
    input_tokens: int
    output_tokens: int  # includes thinking tokens, even when the thinking text isn't shown

    @property
    def cost(self) -> float:
        return cost_of(self.model, self.input_tokens, self.output_tokens)


@dataclass
class CostTracker:
    """Collects every API call of one run, so the total and the breakdown can be shown."""

    calls: list[CallCost] = field(default_factory=list)

    def add(self, response) -> None:
        # response.model is the model that actually answered, which could be a fallback model.
        self.calls.append(
            CallCost(
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        )

    @property
    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.calls)

    @property
    def cost(self) -> float:
        return sum(call.cost for call in self.calls)

    def print_table(self) -> None:
        # Each cell is padded first, then colored, so the invisible color codes don't shift columns.
        print(colors.status(f"  {'call':>4}  {'model':<18} {'stop_reason':<12} {'input':>8} {'output':>8} {'cost':>9}"))
        for number, call in enumerate(self.calls, start=1):
            print(
                f"  {number:>4}  {call.model:<18} {call.stop_reason:<12} "
                f"{call.input_tokens:>8,} {call.output_tokens:>8,} " + colors.score(f"{f'${call.cost:.4f}':>9}")
            )
        print(
            colors.bold(f"  {'total':>4}  {'':<18} {'':<12} {self.input_tokens:>8,} {self.output_tokens:>8,} ")
            + colors.bold(colors.score(f"{f'${self.cost:.4f}':>9}"))
        )
