"""Tests for the Pons solver wrapper."""

import sys
from pathlib import Path
import tempfile
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from env.connect_four_env import ConnectFourEnv
from env.pons_wrapper import PonsSolver, PonsSolverError


def _get_solver():
    """Create a PonsSolver with reduced fallback depth for test speed."""
    return PonsSolver(fallback_depth=4)


def _make_echo_solver(tmp_path):
    input_log = tmp_path / "solver_input.txt"
    solver_path = tmp_path / "connect4_solver"
    solver_path.write_text(
        f"""#!/bin/sh
cat > "{input_log}"
while IFS= read -r line; do
  echo "$line 0"
done < "{input_log}"
""",
        encoding="utf-8",
    )
    solver_path.chmod(0o755)
    book_path = tmp_path / "7x6.book"
    book_path.write_text("stub", encoding="utf-8")
    return solver_path, book_path, input_log


def test_falls_back_when_binary_absent():
    """No error raised, returns minimax result."""
    solver = PonsSolver(solver_path="/nonexistent/binary", fallback_depth=4)
    env = ConnectFourEnv()
    scores = solver.analyze(env)
    assert isinstance(scores, dict)
    assert len(scores) > 0


def test_is_available_false_without_binary():
    """is_available returns False when binary doesn't exist."""
    solver = PonsSolver(solver_path="/nonexistent/binary", fallback_depth=4)
    assert not solver.is_available()


def test_is_available_false_without_book():
    """Binary-only setups should not count as available."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        solver_path = Path(tmp_dir) / "connect4_solver"
        solver_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        solver_path.chmod(0o755)
        solver = PonsSolver(solver_path=str(solver_path), book_path=str(Path(tmp_dir) / "7x6.book"))
        assert not solver.is_available()


def test_strict_raises_when_binary_absent():
    """Strict mode should fail fast instead of silently falling back."""
    solver = PonsSolver(solver_path="/nonexistent/binary", fallback_depth=4, strict=True)
    env = ConnectFourEnv()
    with pytest.raises(PonsSolverError):
        solver.analyze(env)


def test_strict_raises_on_unparseable_solver_output():
    """Malformed solver output should be fatal in strict mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        solver_path = Path(tmp_dir) / "connect4_solver"
        solver_path.write_text("#!/bin/sh\necho junk-output\n", encoding="utf-8")
        solver_path.chmod(0o755)
        book_path = Path(tmp_dir) / "7x6.book"
        book_path.write_text("stub", encoding="utf-8")

        solver = PonsSolver(
            solver_path=str(solver_path),
            book_path=str(book_path),
            fallback_depth=4,
            strict=True,
        )
        env = ConnectFourEnv()
        with pytest.raises(PonsSolverError):
            solver.analyze(env)


def test_nonstrict_still_falls_back_on_unparseable_solver_output():
    """Development mode keeps the old fallback behavior."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        solver_path = Path(tmp_dir) / "connect4_solver"
        solver_path.write_text("#!/bin/sh\necho junk-output\n", encoding="utf-8")
        solver_path.chmod(0o755)
        book_path = Path(tmp_dir) / "7x6.book"
        book_path.write_text("stub", encoding="utf-8")

        solver = PonsSolver(
            solver_path=str(solver_path),
            book_path=str(book_path),
            fallback_depth=4,
            strict=False,
        )
        env = ConnectFourEnv()
        scores = solver.analyze(env)
        assert isinstance(scores, dict)
        assert len(scores) > 0


def test_normalize_reward_in_range():
    """Always [0,1]."""
    solver = _get_solver()
    env = ConnectFourEnv()
    for col in env.legal_moves():
        reward = solver.normalize_reward(env, col)
        assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for col {col}"


def test_normalize_reward_optimal_is_one():
    """Best move gets 1.0."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    best = solver.best_move(env)
    reward = solver.normalize_reward(env, best)
    assert reward == 1.0


def test_analyze_returns_legal_cols_only():
    """All returned columns are legal moves."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    scores = solver.analyze(env)
    legal = set(env.legal_moves())
    for col in scores:
        assert col in legal, f"Column {col} not in legal moves {legal}"


def test_pons_analyze_scores_terminal_children_without_solver_input(tmp_path):
    """Immediate winning children should not be sent to Pons as unfinished states."""
    solver_path, book_path, input_log = _make_echo_solver(tmp_path)
    solver = PonsSolver(
        solver_path=str(solver_path),
        book_path=str(book_path),
        fallback_depth=4,
        strict=True,
    )
    env = ConnectFourEnv()
    env.from_move_sequence([int(ch) for ch in "306345344510533"])

    terminal_col = 6
    next_env = env.copy()
    next_env.make_move(terminal_col)
    assert next_env.is_terminal()

    scores = solver.analyze(env)

    assert set(scores) == set(env.legal_moves())
    assert scores[terminal_col] == PonsSolver.TERMINAL_WIN_SCORE

    pons_base = "".join(str(int(ch) + 1) for ch in env.to_move_sequence())
    assert pons_base + str(terminal_col + 1) not in input_log.read_text().splitlines()


def test_pons_batch_scores_terminal_children(tmp_path):
    """Batched analysis should also preserve all legal columns around terminal children."""
    solver_path, book_path, _ = _make_echo_solver(tmp_path)
    solver = PonsSolver(
        solver_path=str(solver_path),
        book_path=str(book_path),
        fallback_depth=4,
        strict=True,
    )
    env = ConnectFourEnv()
    env.from_move_sequence([int(ch) for ch in "306345344510533"])

    scores = solver.analyze_batch([env])[0]

    assert set(scores) == set(env.legal_moves())
    assert scores[6] == PonsSolver.TERMINAL_WIN_SCORE


def test_optimal_opponent_response_is_legal():
    """Optimal opponent response is a legal move."""
    solver = _get_solver()
    env = ConnectFourEnv()
    env.make_move(3)
    env.make_move(4)
    for col in env.legal_moves():
        opp = solver.optimal_opponent_response(env, col)
        if opp >= 0:  # -1 means terminal
            next_env = env.copy()
            next_env.make_move(col)
            if not next_env.is_terminal():
                assert opp in next_env.legal_moves()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
