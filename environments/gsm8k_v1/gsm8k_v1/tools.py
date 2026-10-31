"""Calculator tool for GSM8K. The model writes calc(expression). Python evaluates it.

Only + - * / are formulas. The reward compares a canonical form of those
formulas with the solution, and the reported score is still the last formula's
value. Gold formulas are not put in the prompt.
"""

from __future__ import annotations

import ast
import re

from gsm8k_v1.rewards import gold_from_solution, normalize_number

TOOL_SYSTEM = (
    "Solve the grade-school math problem. "
    "Write one calc(expression) call for every arithmetic step. "
    "The calculator computes the value. "
    "Put the final number in <answer>...</answer>."
)

# Operation checklist. Used only when scoring asks for it. The weights were
# trained with TOOL_SYSTEM, so this prompt is a measured variant, not the default.
TOOL_CHECKLIST = (
    "Solve the grade-school math problem. "
    "Write one calc(expression) call for every arithmetic step. "
    "Half, split, or a fraction uses division. "
    "A percent uses multiplication by the decimal, so 60% is *0.6. "
    "Combined amounts use addition. "
    "What is left, or a difference, uses subtraction. "
    "Repeated groups use multiplication. "
    "The calculator computes the value. "
    "Put the final number in <answer>...</answer>."
)

_ANN = re.compile(r"<<\s*([^=<>]+?)\s*=\s*([^=<>]+?)\s*>>")
_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)
_PROSE_CHARS = set("0123456789.+-*/(),$%xX× \t") | set("−–—")


def safe_calc(expr: str) -> float:
    """Evaluate arithmetic. Rejects names, calls, and powers."""
    cleaned = _clean_expr(expr)
    if not cleaned or len(cleaned) > 80:
        raise ValueError("empty or oversized expression")
    tree = ast.parse(cleaned, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
    value = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {})  # noqa: S307
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("not a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return float(value)


def format_value(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text


def _clean_expr(expr: str) -> str:
    text = (expr or "").replace(",", "").replace("$", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", "", text)


def iter_calc_exprs(text: str) -> list[str]:
    """Expressions inside calc(), including nested parentheses."""
    found: list[str] = []
    key = "calc("
    i = 0
    while True:
        start = (text or "").find(key, i)
        if start < 0:
            break
        depth = 1
        k = start + len(key)
        while k < len(text) and depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        if depth != 0:
            break
        expr = text[start + len(key) : k - 1].strip()
        if expr and len(expr) <= 80:
            found.append(expr)
        i = k
    return found


def _calls(completion: str) -> list[str]:
    calls = iter_calc_exprs(completion or "")
    if calls:
        return calls
    return [expr for expr, _shown in _ANN.findall(completion or "")]


def calculator_insertion(text: str) -> str | None:
    """Value to append when a calc() just closed and the model has not written '='.

    Returns None if the call is unfinished, already has a value, or is not safe
    arithmetic. The caller must not train on the inserted tokens.
    """
    expr = _trailing_calc(text or "")
    if expr is None:
        return None
    try:
        return f" = {format_value(safe_calc(expr))}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None


def _trailing_calc(text: str) -> str | None:
    key = "calc("
    start = text.rfind(key)
    if start < 0:
        return None
    depth = 1
    k = start + len(key)
    while k < len(text) and depth:
        if text[k] == "(":
            depth += 1
        elif text[k] == ")":
            depth -= 1
        k += 1
    if depth != 0:
        return None
    if text[k:].strip():
        return None
    expr = text[start + len(key) : k - 1].strip()
    if not expr or len(expr) > 80:
        return None
    return expr


def _strip_repeated_value(text: str) -> str:
    """Drop the solution's copy of a number that calc() just stated.

    The marker sits immediately after the calculator value. Parentheses inside
    the call must not end the match, or the value glues to the next digits.
    """
    out: list[str] = []
    i = 0
    while True:
        marker = text.find("\x00", i)
        if marker < 0:
            out.append(text[i:])
            break
        out.append(text[i:marker])
        eq = text.rfind(" = ", i, marker)
        value = text[eq + 3 : marker] if eq >= 0 else ""
        k = marker + 1
        while k < len(text) and text[k] in " \t":
            k += 1
        alternatives = [value]
        if value.startswith("0."):
            alternatives.append(value[1:])
        for alt in alternatives:
            end = k + len(alt)
            if alt and text.startswith(alt, k) and (end >= len(text) or not text[end].isdigit()):
                k = end
                break
        i = k
    return "".join(out)


def tool_trace(solution: str, gold: str) -> str | None:
    """Solution text with each <<expr=result>> replaced by calc(expr) = value.

    Rows whose annotations never reach the official answer are skipped. A
    repaired prose step is a reward target, not a supervised string.
    """
    anns = _ANN.findall(solution or "")
    if not anns:
        return None
    last = None
    for expr, _shown in anns:
        try:
            last = safe_calc(expr)
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            return None
    if last is None or normalize_number(format_value(last)) != normalize_number(str(gold)):
        return None

    def repl(match: re.Match[str]) -> str:
        expr = match.group(1).strip()
        return f"calc({expr}) = {format_value(safe_calc(expr))}\x00"

    text = _ANN.sub(repl, solution or "")
    text = _strip_repeated_value(text)
    text = re.sub(r"\n?####.*$", "", text.strip())
    text = text.rstrip() + f"\n<answer>{normalize_number(str(gold))}</answer>"
    return text


def _number(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return format_value(float(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _number(node.operand)
        if inner is None:
            return None
        if inner.startswith("-"):
            return format_value(-float(inner))
        return format_value(-float(inner))
    return None


def _flatten(node: ast.AST, op: type) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, op):
        return _flatten(node.left, op) + _flatten(node.right, op)
    return [node]


def _form(node: ast.AST) -> tuple[str, str]:
    """('num', text) or ('bin', text). Identity *1, /1, +0, and -0 are removed."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _form(node.operand)
    number = _number(node)
    if number is not None and not isinstance(node, ast.BinOp):
        return ("num", number)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        kind, text = _form(node.operand)
        if kind == "num":
            return ("num", format_value(-float(text)))
        return ("bin", f"(-{text})")
    if not isinstance(node, ast.BinOp):
        raise ValueError("not arithmetic")
    if isinstance(node.op, ast.Add):
        parts = [_form(part) for part in _flatten(node, ast.Add)]
        parts = [part for part in parts if part != ("num", "0")]
        if not parts:
            return ("num", "0")
        if len(parts) == 1:
            return parts[0]
        body = "+".join(sorted(part[1] for part in parts))
        return ("bin", f"({body})")
    if isinstance(node.op, ast.Mult):
        parts = [_form(part) for part in _flatten(node, ast.Mult)]
        parts = [part for part in parts if part != ("num", "1")]
        if not parts:
            return ("num", "1")
        if len(parts) == 1:
            return parts[0]
        body = "*".join(sorted(part[1] for part in parts))
        return ("bin", f"({body})")
    if isinstance(node.op, ast.Sub):
        left = _form(node.left)
        right = _form(node.right)
        if right == ("num", "0"):
            return left
        return ("bin", f"({left[1]}-{right[1]})")
    if isinstance(node.op, ast.Div):
        left = _form(node.left)
        right = _form(node.right)
        if right == ("num", "1"):
            return left
        # 2.5/100 is the same coefficient as 0.025. Fold only a literal percent,
        # not (20*20)/100, so a different operation is not treated as the same formula.
        if right == ("num", "100") and left[0] == "num":
            return ("num", format_value(float(left[1]) / 100.0))
        return ("bin", f"({left[1]}/{right[1]})")
    raise ValueError("operator")


def canonicalize(expr: str) -> str | None:
    """Canonical formula, or None for a bare number, an identity, or unsafe syntax.

    Addition and multiplication are reordered. 0.5 and .5 match. A literal
    divided by 100 matches that percent written as a decimal. 20+4 does not
    match 48/2, because the operations are different.
    """
    cleaned = _clean_expr(expr)
    if not cleaned or len(cleaned) > 80:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
        kind, text = _form(tree.body)
    except (SyntaxError, ValueError, OverflowError):
        return None
    if kind != "bin":
        return None
    return text


def gold_expressions(solution: str) -> list[str]:
    """Annotated formulas that are safe arithmetic, in solution order."""
    found: list[str] = []
    for expr, _shown in _ANN.findall(solution or ""):
        canon = canonicalize(expr)
        if canon:
            found.append(canon)
    return found


def _prose_equations(solution: str) -> list[str]:
    """Arithmetic written outside << >> whose computed value matches the number after '='."""
    text = _ANN.sub(" ", solution or "").split("####")[0]
    found: list[str] = []
    for index, char in enumerate(text):
        if char != "=":
            continue
        left_end = index - 1
        while left_end >= 0 and text[left_end] in _PROSE_CHARS:
            left_end -= 1
        left = _clean_expr(text[left_end + 1 : index]).replace("%", "").replace("×", "*")
        left = re.sub(r"(?<=[\d)])[xX](?=[\d.(])", "*", left)
        right_at = index + 1
        while right_at < len(text) and text[right_at] in " $":
            right_at += 1
        shown = re.match(r"(\.\d+|\d[\d,]*(?:\.\d+)?)", text[right_at:])
        if not left or not shown or not any(op in left for op in "+-*/"):
            continue
        try:
            value = format_value(safe_calc(left))
        except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            continue
        if normalize_number(value) != normalize_number(shown.group(1)):
            continue
        found.append(left)
    return found


_LINEAR = re.compile(
    r"(?<![A-Za-z0-9])\(?"
    r"(\d*\.\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?)"
    r"\)?\s*\*?\s*"
    r"([A-Za-z])"
    r"\s*=\s*\$?\s*"
    r"(\d[\d,]*(?:\.\d+)?)"
    r"(?![A-Za-z0-9])"
)
_SOLVED = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])\s*=\s*\$?"
    r"(\d[\d,]*(?:\.\d+)?)\s*/\s*\(?"
    r"(\d*\.\d+|\d+\s*/\s*\d+|\d+(?:\.\d+)?)"
    r"\)?(?![A-Za-z0-9])"
)


def _linear_answers(solution: str, target: str) -> list[str]:
    """Arithmetic for a one-letter equation whose solution is the official answer.

    `2/3 * x = 12` becomes `12/(2/3)`. `.75X = 19.50` becomes `19.50/.75`.
    A bare `x = 18` is not a formula. The value has to equal the official answer.
    """
    text = (solution or "").split("####")[0]
    found: list[str] = []

    seen: set[str] = set()

    def _keep(expr: str) -> None:
        canon = canonicalize(expr)
        if not canon or canon in seen:
            return
        if _formula_value(canon) == target:
            seen.add(canon)
            found.append(expr)

    for coeff, _var, rhs in _LINEAR.findall(text):
        _keep(f"({rhs.replace(',', '')})/({re.sub(r'\s+', '', coeff)})")
    for _var, rhs, coeff in _SOLVED.findall(text):
        _keep(f"({rhs.replace(',', '')})/({re.sub(r'\s+', '', coeff)})")
    return found


def reference_expressions(solution: str, gold: str | None = None) -> list[str]:
    """Formulas the reward may copy. Empty when they do not end at the official answer.

    Annotated steps are kept. A missing last step is taken from the prose, or
    from a one-letter equation, only when that formula evaluates to the ####
    number. Otherwise the row is not an expression target.
    """
    target = normalize_number(str(gold)) if gold is not None else gold_from_solution(solution or "")
    formulas = gold_expressions(solution)
    seen = set(formulas)
    for expr in _prose_equations(solution) + _linear_answers(solution, target):
        canon = canonicalize(expr)
        if canon and canon not in seen:
            formulas.append(canon)
            seen.add(canon)
    # An earlier formula can be the chosen answer when a later one is another branch.
    # Move that formula to the end. Do not invent a formula that was never written.
    if formulas and _formula_value(formulas[-1]) != target:
        for index, canon in enumerate(formulas):
            if _formula_value(canon) == target:
                formulas = formulas[:index] + formulas[index + 1 :] + [canon]
                break
    if formulas and _formula_value(formulas[-1]) == target:
        return formulas
    return []


def _formula_value(canon: str) -> str:
    try:
        return normalize_number(format_value(safe_calc(canon)))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return ""


def calc_spans(text: str) -> list[tuple[int, int, str | None]]:
    """Each calc() as (start, end, canonical formula or None). End is past the ')'."""
    spans: list[tuple[int, int, str | None]] = []
    key = "calc("
    i = 0
    source = text or ""
    while True:
        start = source.find(key, i)
        if start < 0:
            break
        depth = 1
        k = start + len(key)
        while k < len(source) and depth:
            if source[k] == "(":
                depth += 1
            elif source[k] == ")":
                depth -= 1
            k += 1
        if depth != 0:
            break
        spans.append((start, k, canonicalize(source[start + len(key) : k - 1])))
        i = k
    return spans


def ordered_prefix_len(completion: str, solution: str, gold: str | None = None) -> int:
    """How many leading reference formulas were copied, in order."""
    refs = reference_expressions(solution, gold)
    got = 0
    for pred in predicted_formulas(completion):
        if got >= len(refs) or pred != refs[got]:
            break
        got += 1
    return got


def prefix_end_after(text: str, solution: str, gold: str | None, steps: int) -> int | None:
    """Character end of the first `steps` reference formulas, in order."""
    if steps <= 0:
        return 0
    refs = reference_expressions(solution, gold)
    got = 0
    end = None
    for _start, stop, canon in calc_spans(text):
        if canon and got < len(refs) and canon == refs[got]:
            got += 1
            end = stop
            if got >= steps:
                return end
            continue
        break
    return None


def prefix_end(text: str, solution: str, gold: str | None = None) -> int | None:
    """Character end of the last calc() that still follows the reference, in order."""
    refs = reference_expressions(solution, gold)
    got = 0
    end = None
    for start, stop, canon in calc_spans(text):
        if canon and got < len(refs) and canon == refs[got]:
            got += 1
            end = stop
            continue
        break
    return end


def diverge_span(text: str, solution: str, gold: str | None = None) -> tuple[int, int] | None:
    """The first calc() that leaves the reference prefix."""
    refs = reference_expressions(solution, gold)
    got = 0
    for start, stop, canon in calc_spans(text):
        if canon and got < len(refs) and canon == refs[got]:
            got += 1
            continue
        return (start, stop)
    return None


def answer_formula_span(text: str, solution: str, gold: str | None = None) -> tuple[int, int] | None:
    """Span of the official answer formula when a later calc() continues past it.

    The match is the reference formula, not an earlier number that happens to
    equal the answer. None when that formula is missing or already last.
    """
    refs = reference_expressions(solution, gold)
    if not refs:
        return None
    answer = refs[-1]
    spans = calc_spans(text)
    for index, (start, stop, canon) in enumerate(spans):
        if canon == answer and index < len(spans) - 1:
            return (start, stop)
    return None


def repeat_start(text: str) -> int | None:
    """Start of the first calc() that repeats the canonical formula before it."""
    spans = calc_spans(text)
    for index in range(1, len(spans)):
        current = spans[index][2]
        if current and current == spans[index - 1][2]:
            return spans[index][0]
    return None


def repeated_formula(text: str) -> bool:
    """True when the last formula is the same as the one before it."""
    formulas = predicted_formulas(text)
    return len(formulas) >= 2 and formulas[-1] == formulas[-2]


def cut_repeated_tail(text: str) -> str:
    """Drop the answer from the start of a formula that repeats the one before it.

    The repeated call is not a new step. Keeping it, and whatever follows, changes
    the last formula without adding information.
    """
    positions: list[tuple[int, str | None]] = []
    key = "calc("
    i = 0
    while True:
        start = text.find(key, i)
        if start < 0:
            break
        depth = 1
        k = start + len(key)
        while k < len(text) and depth:
            if text[k] == "(":
                depth += 1
            elif text[k] == ")":
                depth -= 1
            k += 1
        if depth != 0:
            break
        positions.append((start, canonicalize(text[start + len(key) : k - 1])))
        i = k
    for n in range(1, len(positions)):
        current = positions[n][1]
        if current and current == positions[n - 1][1]:
            return text[: positions[n][0]].rstrip()
    return text


def predicted_formulas(completion: str) -> list[str]:
    found: list[str] = []
    for expr in _calls(completion):
        canon = canonicalize(expr)
        if canon:
            found.append(canon)
    return found


def last_formula_value(completion: str) -> str:
    """Value of the last real formula. A bare number or an identity is ignored."""
    formulas = predicted_formulas(completion)
    if not formulas:
        return ""
    return format_value(safe_calc(formulas[-1]))


def _f1(gold: set[str], pred: set[str]) -> float:
    if not gold or not pred:
        return 0.0
    hit = len(gold & pred)
    if hit == 0:
        return 0.0
    precision = hit / len(pred)
    recall = hit / len(gold)
    return 2 * precision * recall / (precision + recall)


def expression_f1(completion: str, solution: str, question: str = "") -> float:
    """Overlap of canonical formulas. The question is not an allow-list."""
    del question
    gold = reference_expressions(solution)
    return _f1(set(gold), set(predicted_formulas(completion)))


def tool_match(completion: str, gold: str) -> float:
    """1 when the last real formula equals gold. calc(72) and calc(72*1) do not."""
    got = last_formula_value(completion)
    if not got:
        return 0.0
    return 1.0 if normalize_number(got) == normalize_number(str(gold)) else 0.0


def tool_shape(completion: str, solution: str, question: str = "") -> float:
    """Fraction of reference steps shown. Extra calls only reduce that fraction."""
    del question
    n_gold = len(reference_expressions(solution))
    n_formulas = len(predicted_formulas(completion))
    n_calls = len(_calls(completion))
    if n_gold == 0 or n_formulas == 0:
        return 0.0
    # Cap the fraction at 1. A one-step answer with many extra formulas used to
    # score above the perfect trace, because the ratio grew faster than the penalty.
    shown = min(1.0, n_formulas / n_gold)
    extra = max(0, n_calls - n_gold)
    return max(0.0, shown - 0.5 * extra)


def tool_reward(completion: str, gold: str, solution: str, question: str = "") -> float:
    """Finished answers use 0.25*match + 0.25*match*overlap + 0.35*overlap + 0.15*shape.

    An unfinished answer scores 0.04 per leading reference step, and never more
    than 0.24, so a correct last formula still ranks above a longer wrong one.
    A right number from the wrong formula contributes 0.25 from the match term.
    """
    formulas = expression_f1(completion, solution, question)
    matched = tool_match(completion, gold)
    if matched:
        final = 0.5 + 0.5 * formulas
        shape = tool_shape(completion, solution, question)
        return 0.5 * final + 0.35 * formulas + 0.15 * shape
    steps = ordered_prefix_len(completion, solution, gold)
    return min(0.24, 0.04 * steps)


def tool_calls(completion: str) -> int:
    return len(_calls(completion))
