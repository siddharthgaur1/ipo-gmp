"""Streamlit derives a widget's id from its type, label, options and default.

Two widgets sharing all of those collide, and the whole tab dies at runtime with
StreamlitDuplicateElementId - which is what happened to the Dashboard and IPO
Table tabs, both of which had:

    st.multiselect("Category", ["Mainboard", "SME"], default=["Mainboard", "SME"])

Nothing catches that at import time, so this test does: within a single widget
type, any label used more than once must carry an explicit `key=`.

Grouping is per (widget_type, label) on purpose - `selectbox("Category")` and
`multiselect("Category")` get different ids and do not collide.
"""

import ast
from collections import defaultdict
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "src" / "app.py"
WIDGETS = {"multiselect", "selectbox", "text_input", "slider", "radio",
           "number_input", "checkbox", "text_area", "date_input"}


def test_repeated_widget_labels_have_explicit_keys():
    tree = ast.parse(APP.read_text(encoding="utf-8"))

    uses: dict[tuple[str, str], list[tuple[int, bool]]] = defaultdict(list)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        kind = node.func.attr
        if kind not in WIDGETS or not node.args:
            continue
        label = node.args[0]
        if not (isinstance(label, ast.Constant) and isinstance(label.value, str)):
            continue
        has_key = any(kw.arg == "key" for kw in node.keywords)
        uses[(kind, label.value)].append((node.lineno, has_key))

    offenders = {
        f"{kind}({label!r})": [ln for ln, has_key in seen if not has_key]
        for (kind, label), seen in uses.items()
        if len(seen) > 1 and any(not has_key for _, has_key in seen)
    }
    assert not offenders, (
        "widget labels reused within one widget type without an explicit key= "
        f"(raises StreamlitDuplicateElementId at runtime): {offenders}"
    )
