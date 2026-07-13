"""알파 수식 문자열을 field 패널에 대해 '안전하게' 평가한다.

"rank(neg(ts_delta(close, 5)))" 같은 수식을 파이썬 AST 로 파싱한 뒤 직접 순회한다.
허용하는 노드는 극소수 화이트리스트뿐:
  * Name      -> field 패널(close, quote_volume ...) 또는 연산자 이름
  * Call      -> operator(args...)
  * Constant  -> int/float/bool 리터럴
  * BinOp     -> + - * /
  * UnaryOp   -> 단항 -, +

그 외(속성 접근, 인덱싱, 람다, 임포트 등)는 전부 에러.
그래서 eval() 을 쓰지 않는다 — eval 은 파이썬 런타임 전체를 노출시킨다.

공개 API:
    evaluate(expression, panels) -> pd.DataFrame  (알파 원점수 패널)
"""
from __future__ import annotations

import ast

from src.backtest.operators import OPERATORS

_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}


class ExpressionError(ValueError):
    """잘못됐거나 허용되지 않은 수식."""


def evaluate(expression: str, panels: dict):
    if not isinstance(expression, str) or not expression.strip():
        raise ExpressionError("expression 은 비어있지 않은 문자열이어야 함")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"수식 파싱 실패 {expression!r}: {exc}") from exc
    return _eval_node(tree.body, panels)


def _eval_node(node, panels):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ExpressionError(f"숫자 리터럴만 허용, got {node.value!r}")

    if isinstance(node, ast.Name):
        return _resolve_name(node.id, panels)

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, panels)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ExpressionError(f"단항연산자 {type(node.op).__name__} 불가")

    if isinstance(node, ast.BinOp):
        op_name = _BINOPS.get(type(node.op))
        if op_name is None:
            raise ExpressionError(
                f"이항연산자 {type(node.op).__name__} 불가 (+ - * / 만)"
            )
        left = _eval_node(node.left, panels)
        right = _eval_node(node.right, panels)
        return OPERATORS[op_name](left, right)

    if isinstance(node, ast.Call):
        if node.keywords:
            raise ExpressionError("수식에 키워드 인자 불가")
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("연산자 직접 호출만 허용")
        fname = node.func.id
        fn = OPERATORS.get(fname)
        if fn is None:
            raise ExpressionError(f"알 수 없는 연산자 {fname!r}")
        args = [_eval_node(a, panels) for a in node.args]
        try:
            return fn(*args)
        except TypeError as exc:
            raise ExpressionError(f"{fname}() 인자 오류: {exc}") from exc

    raise ExpressionError(f"허용되지 않은 문법: {type(node).__name__}")


def _resolve_name(name, panels):
    if name in panels:
        return panels[name]
    if name in OPERATORS:
        raise ExpressionError(f"{name!r} 는 연산자 — 호출해야 함: {name}(...)")
    raise ExpressionError(f"알 수 없는 field {name!r}; 사용가능: {sorted(panels)}")


def required_fields(expression):
    """수식에서 쓰인 field 이름들(연산자 제외)을 뽑는다. panel 로딩에 사용."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"수식 파싱 실패 {expression!r}: {exc}") from exc
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in OPERATORS:
            names.add(node.id)
    return names


# neutralization 모드가 수식 밖에서 추가로 요구하는 패널 필드.
# 예: turnover_rank 는 유동성 버킷을 만들려고 quote_volume 을 별도로 읽는다.
# 이걸 반영하지 않으면 패널 로더가 quote_volume 을 안 실어서 엔진이 런타임에 죽는다.
_NEUTRALIZATION_FIELDS = {
    "turnover_rank": {"quote_volume"},
}


def spec_required_fields(expression, neutralization="none"):
    """수식 필드 + neutralization 이 추가로 요구하는 필드의 합집합.

    패널 로딩·백테스트 가능여부 판정은 항상 이걸 써야 한다 — 수식만 보면
    turnover_rank 처럼 중립화 단계에서 별도 필드를 읽는 알파를 놓친다."""
    return set(required_fields(expression)) | _NEUTRALIZATION_FIELDS.get(neutralization, set())
