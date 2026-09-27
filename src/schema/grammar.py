"""Optional context-free grammar constraints for structured generation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

try:
    from lark import Lark, UnexpectedInput
except ImportError:  # pragma: no cover
    Lark = None
    UnexpectedInput = Exception


class GrammarUnavailable(RuntimeError):
    pass


class GrammarValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GrammarSpec:
    grammar: str
    start: str = "start"


class GrammarConstraint:
    """Compile and validate an EBNF grammar without changing generation APIs."""

    def __init__(self, spec: GrammarSpec) -> None:
        if not spec.grammar.strip():
            raise GrammarValidationError("grammar cannot be empty")
        if Lark is None:
            raise GrammarUnavailable("lark is required for grammar constraints")
        try:
            self._parser = Lark(spec.grammar, parser="lalr", start=spec.start)
        except Exception as exc:
            raise GrammarValidationError(str(exc)) from exc
        self.spec = spec

    def validate(self, text: str) -> bool:
        try:
            self._parser.parse(text)
            return True
        except UnexpectedInput as exc:
            raise GrammarValidationError(str(exc)) from exc

    def accepts(self, text: str) -> bool:
        try:
            self._parser.parse(text)
            return True
        except UnexpectedInput:
            return False

    def filter_candidates(self, prefix: str, candidates: Iterable[str]) -> list[str]:
        """Return candidates that preserve at least one parseable continuation.

        The method is deliberately conservative: a candidate is retained if the
        resulting prefix is either a complete parse or still a valid parser prefix.
        """
        accepted: list[str] = []
        for candidate in candidates:
            value = prefix + candidate
            if self.accepts(value):
                accepted.append(candidate)
                continue
            # Lark's interactive parser provides exact token expectations without
            # inventing model probabilities. Character-level filtering is therefore
            # intentionally left to the tokenizer integration layer.
            try:
                interactive = self._parser.parse_interactive(value)
                list(interactive.accepts())
                accepted.append(candidate)
            except Exception:
                pass
        return accepted
