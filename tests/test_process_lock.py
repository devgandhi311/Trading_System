from __future__ import annotations

import json
import os

import pytest

import runtime.process_lock as process_lock
from runtime.process_lock import PaperProcessLock, ProcessLockError


def test_second_process_owner_is_rejected(tmp_path):
    path = tmp_path / "paper.lock"
    first = PaperProcessLock(path)
    first.acquire()
    try:
        with pytest.raises(ProcessLockError, match="already active"):
            PaperProcessLock(path).acquire()
    finally:
        first.release()


def test_stale_local_lock_is_recovered(tmp_path, monkeypatch):
    path = tmp_path / "paper.lock"
    path.write_text(json.dumps({"pid": 99999999, "hostname": __import__("socket").gethostname()}))
    monkeypatch.setattr(process_lock, "_is_process_alive", lambda _pid: False)
    with PaperProcessLock(path):
        assert json.loads(path.read_text())["pid"] == os.getpid()
    assert not path.exists()


def test_malformed_lock_fails_closed(tmp_path):
    path = tmp_path / "paper.lock"
    path.write_text("not-json")
    with pytest.raises(ProcessLockError, match="Malformed"):
        PaperProcessLock(path).acquire()


@pytest.mark.skipif(os.name != "nt", reason="Windows API behavior")
def test_windows_current_pid_is_active_without_os_kill(monkeypatch):
    monkeypatch.setattr(os, "kill", lambda *_: pytest.fail("os.kill must not be called on Windows"))
    assert process_lock._is_process_alive(os.getpid()) is True


@pytest.mark.skipif(os.name != "nt", reason="Windows API behavior")
def test_windows_invalid_parameter_pid_is_stale(monkeypatch):
    monkeypatch.setattr(process_lock, "_windows_process_api", lambda: (
        lambda *_: 0, pytest.fail, pytest.fail, lambda: process_lock._ERROR_INVALID_PARAMETER,
    ))
    assert process_lock._is_process_alive(99999999) is False


@pytest.mark.skipif(os.name != "nt", reason="Windows API behavior")
def test_windows_access_denied_is_active_unverifiable(monkeypatch):
    monkeypatch.setattr(process_lock, "_windows_process_api", lambda: (
        lambda *_: 0, pytest.fail, pytest.fail, lambda: process_lock._ERROR_ACCESS_DENIED,
    ))
    assert process_lock._is_process_alive(12345) is True


@pytest.mark.skipif(os.name != "nt", reason="Windows API behavior")
@pytest.mark.parametrize("exit_code", [process_lock._STILL_ACTIVE, 0])
def test_windows_process_handle_is_always_closed(monkeypatch, exit_code):
    closed = []

    def get_exit_code(_handle, output):
        output._obj.value = exit_code
        return 1

    monkeypatch.setattr(process_lock, "_windows_process_api", lambda: (
        lambda *_: 101, get_exit_code, lambda handle: closed.append(handle) or 1, lambda: 0,
    ))
    assert process_lock._is_process_alive(12345) is (exit_code == process_lock._STILL_ACTIVE)
    assert closed == [101]


@pytest.mark.skipif(os.name != "nt", reason="Windows API behavior")
def test_windows_query_failure_fails_closed_and_closes_handle(monkeypatch):
    closed = []
    monkeypatch.setattr(process_lock, "_windows_process_api", lambda: (
        lambda *_: 202, lambda *_: 0, lambda handle: closed.append(handle) or 1, lambda: 31,
    ))
    with pytest.raises(ProcessLockError, match="Cannot query"):
        process_lock._is_process_alive(12345)
    assert closed == [202]
