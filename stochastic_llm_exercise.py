"""A tiny exercise about why LLM output is stochastic.

This is not a real neural network. Its table of logits stands in for an LLM's
learned next-token scores. Your job is to turn scores into probabilities and
sample from them.

Run: python stochastic_llm_exercise.py
Fill the three TODOs until all checks pass and several sentences print.
"""

import math
import random


# Each token maps to possible next tokens and their unnormalized scores (logits).
MODEL = {
    "<START>": {"robots": 2.0, "language": 1.5, "learning": 1.0},
    "robots": {"can": 2.0, "sometimes": 1.0},
    "language": {"can": 1.0, "is": 2.0},
    "learning": {"is": 2.0, "can": 1.0},
    "can": {"surprise": 2.0, "change": 1.5, "help": 1.0},
    "sometimes": {"surprise": 2.0, "help": 1.0},
    "is": {"probabilistic": 2.5, "useful": 1.0},
    "surprise": {"us": 2.0, "you": 1.0},
    "change": {"with": 2.0, "quickly": 1.0},
    "help": {"us": 1.0, "you": 2.0},
    "probabilistic": {"<END>": 1.0},
    "useful": {"<END>": 1.0},
    "us": {"<END>": 1.0},
    "you": {"<END>": 1.0},
    "with": {"context": 2.0, "temperature": 1.0},
    "quickly": {"<END>": 1.0},
    "context": {"<END>": 1.0},
    "temperature": {"<END>": 1.0},
}


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    """Convert logits to probabilities.

    TODO 1:
    - Reject temperature <= 0 with ValueError.
    - Divide each logit by temperature.
    - For numerical stability, subtract the largest scaled logit.
    - Exponentiate with math.exp, then normalize so the values sum to 1.
    """
    raise NotImplementedError("TODO 1")


def sample_next(options: dict[str, float], temperature: float) -> str:
    """Sample one token from a {token: logit} mapping.

    TODO 2: Use softmax(...) and random.choices(..., k=1).
    Hint: list(options) gives tokens in the same order as options.values().
    """
    raise NotImplementedError("TODO 2")


def generate(temperature: float = 1.0, max_tokens: int = 12) -> str:
    """Generate tokens until <END> or max_tokens is reached.

    TODO 3:
    - Begin at "<START>" with an empty output list.
    - Repeatedly sample from MODEL[current].
    - Stop without appending if the sampled token is "<END>".
    - Otherwise append it and make it the current token.
    - Return the output joined with spaces.
    """
    raise NotImplementedError("TODO 3")


def checks() -> None:
    """Small checks to tell you when the exercise is complete."""
    assert softmax([0.0, 0.0]) == [0.5, 0.5]
    probabilities = softmax([1.0, 2.0, 3.0])
    assert math.isclose(sum(probabilities), 1.0)
    assert probabilities[2] > probabilities[1] > probabilities[0]

    # Lower temperature should concentrate more probability on the top choice.
    cold = softmax([1.0, 2.0], temperature=0.2)
    hot = softmax([1.0, 2.0], temperature=2.0)
    assert cold[1] > hot[1]

    try:
        softmax([1.0], temperature=0.0)
        raise AssertionError("temperature <= 0 should raise ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    checks()
    random.seed(7)  # Repeatable lesson; remove this line for fresh randomness.

    for temperature in (0.2, 1.0, 2.0):
        print(f"\ntemperature={temperature}")
        for _ in range(5):
            print("  " + generate(temperature))

    print("\nNotice: low temperature favors likely tokens; high temperature explores.")
