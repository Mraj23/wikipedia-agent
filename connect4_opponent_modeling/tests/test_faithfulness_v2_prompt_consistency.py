"""Guard the byte-identical prompt code in faithfulness_v2.

`generate_pool.py`, `train_move_only.py`, and `eval_move_quality.py` each
carry their own copy of `SYSTEM_PROMPT`, `_render_board`, `make_messages`,
and `parse_column`. The duplication is intentional — every script is
independently auditable — but it must stay in sync, or the entropy pool
will reflect a different prompt than the trainer ever sees, and held-out
eval will score yet a third question.

This test fails loudly if any of the four contracts drifts.
"""

from __future__ import annotations

import pytest

from env.connect_four_env import ConnectFourEnv
from faithfulness_v2 import eval_move_quality as em
from faithfulness_v2 import generate_pool as gp
from faithfulness_v2 import train_move_only as tm

MODULES = (gp, tm, em)


def _env_from(moves):
    env = ConnectFourEnv()
    for m in moves:
        env.make_move(m)
    return env


def test_system_prompt_byte_identical():
    prompts = {m.__name__: m.SYSTEM_PROMPT for m in MODULES}
    assert len(set(prompts.values())) == 1, prompts


@pytest.mark.parametrize(
    "history",
    [
        [],
        [3, 3],
        [3, 4, 3, 4, 3],
        [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5],
    ],
    ids=["empty", "open", "midgame", "long"],
)
def test_render_board_byte_identical(history):
    env = _env_from(history)
    rendered = {m.__name__: m._render_board(env) for m in MODULES}
    assert len(set(rendered.values())) == 1, rendered


@pytest.mark.parametrize(
    "history",
    [
        [],
        [3, 3],
        [3, 4, 3, 4, 3],
        [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5],
    ],
    ids=["empty", "open", "midgame", "long"],
)
def test_make_messages_identical(history):
    env = _env_from(history)
    messages_per_module = [m.make_messages(env) for m in MODULES]
    first = messages_per_module[0]
    for other in messages_per_module[1:]:
        assert other == first


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3", 3),
        ("  3  ", 3),
        ("Column 5", 5),
        ("I'd play 6 because", 6),
        ("abc", None),
        ("", None),
        ("12345", 1),                # first digit
        ("9", None),                 # out of [0,6] range
        ("I'd play 7 because", None),  # 7 is out of [0,6]; first match comes from "because" -> no match
        ("8 then 4", 4),
    ],
)
def test_parse_column_identical(text, expected):
    results = [m.parse_column(text) for m in MODULES]
    assert all(r == expected for r in results), (text, results)


def test_column_regex_identical():
    patterns = {m.__name__: m._COLUMN_RE.pattern for m in MODULES}
    assert len(set(patterns.values())) == 1, patterns


def test_modules_export_required_symbols():
    """Catch accidental rename of any of the four contract symbols."""
    required = ("SYSTEM_PROMPT", "_render_board", "make_messages", "parse_column", "_COLUMN_RE")
    for module in MODULES:
        for name in required:
            assert hasattr(module, name), f"{module.__name__} missing {name}"
