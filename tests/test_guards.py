"""The two repository guards: commit identities and the private denylist."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


identity = _load("check_identity")
content = _load("check_content")


def test_only_noreply_identities_pass():
    ok = [
        "p4ik <78314794+p4ik@users.noreply.github.com>",
        "GitHub <noreply@github.com>",
        "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
    ]
    assert identity.check(ok) == []
    assert identity.check(["someone <someone@example.com>"]) == [
        "someone <someone@example.com>"
    ]
    assert identity.check(["x <a@users.noreply.github.com.evil.org>"])
    msg = "fix\n\nSigned-off-by: Someone <someone@example.com>\n"
    assert identity.TRAILER.findall(msg) == ["Someone <someone@example.com>"]


def test_patterns_come_from_the_environment_or_a_file_outside_the_tree(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("LEAK_PATTERNS", raising=False)
    monkeypatch.setenv("MLX_BEAM_DENYLIST", str(tmp_path / "missing"))
    assert content.load_patterns() is None
    (tmp_path / "list").write_text("# comment\n\nsecret\\d+\n")
    monkeypatch.setenv("MLX_BEAM_DENYLIST", str(tmp_path / "list"))
    [p] = content.load_patterns()
    assert p.search("SECRET42")
    monkeypatch.setenv("LEAK_PATTERNS", "")
    assert content.load_patterns() == []  # configured, nothing on it


def test_files_are_reported_by_place_and_count_never_by_content(tmp_path, capsys):
    patterns = [content.re.compile("forbidden", content.re.I)]
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing here\n")
    dirty = tmp_path / "dirty.txt"
    dirty.write_text("Forbidden once, forbidden twice\n")
    named = tmp_path / "forbidden.txt"
    named.write_text("")
    found = content.scan_files([str(clean), str(dirty), str(named)], patterns)
    assert found == [(str(dirty), 2), ("a path", 1)]


def test_no_denylist_fails_unless_told_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("LEAK_PATTERNS", raising=False)
    monkeypatch.setenv("MLX_BEAM_DENYLIST", str(tmp_path / "missing"))
    monkeypatch.delenv("LEAK_PATTERNS_REQUIRED", raising=False)
    assert content.main(["check_content.py", str(tmp_path)]) == 1
    monkeypatch.setenv("LEAK_PATTERNS_REQUIRED", "0")
    assert content.main(["check_content.py", str(tmp_path)]) == 0


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.name", "p4ik")
    _git(tmp_path, "config", "user.email", "1+p4ik@users.noreply.github.com")
    (tmp_path / "a.txt").write_text("hello\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_a_range_catches_a_word_added_and_removed_again(repo, monkeypatch):
    monkeypatch.setenv("LEAK_PATTERNS", "forbidden")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "b.txt").write_text("forbidden\n")
    _git(repo, "add", "b.txt")
    _git(repo, "commit", "-q", "-m", "add")
    _git(repo, "rm", "-q", "b.txt")
    _git(repo, "commit", "-q", "-m", "remove again")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    found = content.scan_range(f"{base}..{head}", content.load_patterns())
    assert len(found) == 1 and found[0][1] == 1  # the commit that added it
    assert found[0][0].startswith("commit ")


def test_a_range_catches_the_message_and_the_tree(repo, monkeypatch):
    monkeypatch.setenv("LEAK_PATTERNS", "forbidden")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "c.txt").write_text("clean\n")
    _git(repo, "add", "c.txt")
    _git(repo, "commit", "-q", "-m", "mention the forbidden word")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    found = content.scan_range(f"{base}..{head}", content.load_patterns())
    assert [n for _, n in found] == [1]
    (repo / "a.txt").write_text("forbidden in the tree now\n")
    _git(repo, "commit", "-q", "-am", "edit")
    head2 = _git(repo, "rev-parse", "HEAD")
    found = content.scan_range(f"{head}..{head2}", content.load_patterns())
    assert [n for _, n in found] == [1, 1]  # the commit's addition, the tree
    assert found[1][0] == "a.txt"


def test_a_first_push_scans_the_head_alone(repo, monkeypatch):
    monkeypatch.setenv("LEAK_PATTERNS", "forbidden")
    head = _git(repo, "rev-parse", "HEAD")
    monkeypatch.chdir(repo)
    assert content.scan_range(f"{'0' * 40}..{head}", content.load_patterns()) == []


def test_the_tools_run_as_scripts(repo, monkeypatch):
    monkeypatch.setenv("LEAK_PATTERNS", "forbidden")
    head = _git(repo, "rev-parse", "HEAD")
    r = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_content.py"),
            "--range",
            f"{head}~0..{head}",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    (repo / "a.txt").write_text("forbidden\n")
    r = subprocess.run(
        [sys.executable, str(TOOLS / "check_content.py"), "a.txt"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1
    assert "forbidden" not in r.stderr and "a.txt" in r.stderr


def test_identity_range_with_a_zero_base_checks_the_head_alone(repo, monkeypatch):
    head = _git(repo, "rev-parse", "HEAD")
    r = subprocess.run(
        [
            sys.executable,
            str(TOOLS / "check_identity.py"),
            "--range",
            f"{'0' * 40}..{head}",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
