import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_provenance import environment, file_sha256, git_state, write_run_manifest


def test_file_sha256_matches_a_known_digest(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"abc")
    assert file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_git_state_reports_commit_branch_and_uncommitted_files_for_a_checkout():
    state = git_state(Path(__file__).resolve().parents[1])
    assert state["commit"] and len(state["commit"]) == 40
    assert isinstance(state["uncommitted_files"], list)


def test_git_state_for_a_directory_that_is_not_a_checkout_says_so(tmp_path):
    state = git_state(tmp_path)
    assert state["commit"] is None and state["uncommitted_files"] is None


def test_uncommitted_files_are_listed_so_a_modified_run_is_visible(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "c"], check=True)
    assert git_state(tmp_path)["uncommitted_files"] == []
    (tmp_path / "a.py").write_text("x = 2\n")
    assert git_state(tmp_path)["uncommitted_files"] == ["a.py"]


def test_environment_records_machine_and_package_versions():
    env = environment()
    assert env["machine"] and env["python"]
    assert env["packages"]["numpy"] is not None


def test_manifest_records_settings_and_hashes_and_tolerates_a_missing_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_bytes(b"abc")
    manifest = write_run_manifest(tmp_path / "m.json", tmp_path, {"seed": 123}, {"data": f, "gone": tmp_path / "nope.csv"})
    saved = json.loads((tmp_path / "m.json").read_text())
    assert saved["settings"] == {"seed": 123}
    assert saved["sha256"]["data"] == file_sha256(f) and saved["sha256"]["gone"] is None
    assert saved == json.loads(json.dumps(manifest))


def test_a_git_state_captured_earlier_is_recorded_unchanged(tmp_path):
    earlier = {"commit": "a" * 40, "branch": "b", "uncommitted_files": []}
    manifest = write_run_manifest(tmp_path / "m.json", tmp_path, {}, {}, git=earlier)
    assert manifest["git"] == earlier
