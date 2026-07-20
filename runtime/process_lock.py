"""Single-instance lock for one PAPER state directory."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import socket
from typing import Callable

from utils.time_utils import now_ist


class ProcessLockError(RuntimeError):
    """Raised when exclusive PAPER process ownership cannot be established."""


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


def _windows_process_api() -> tuple[Callable[..., int], Callable[..., int], Callable[..., int], Callable[[], int]]:
    """Return configured kernel32 process-query functions."""

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    get_exit_code = kernel32.GetExitCodeProcess
    get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    get_exit_code.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    return open_process, get_exit_code, close_handle, ctypes.get_last_error


def _is_process_alive(pid: int) -> bool:
    """Return PID liveness without delivering signals on Windows."""

    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ProcessLockError(f"Invalid process lock PID: {pid!r}")

    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            raise ProcessLockError(f"Cannot verify PAPER PID {pid}") from exc
        return True

    open_process, get_exit_code, close_handle, get_last_error = _windows_process_api()
    handle = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        error = get_last_error()
        if error == _ERROR_INVALID_PARAMETER:
            return False
        if error == _ERROR_ACCESS_DENIED:
            return True
        raise ProcessLockError(f"Cannot verify PAPER PID {pid}: Windows error {error}")

    try:
        exit_code = wintypes.DWORD()
        if not get_exit_code(handle, ctypes.byref(exit_code)):
            error = get_last_error()
            if error == _ERROR_ACCESS_DENIED:
                return True
            raise ProcessLockError(f"Cannot query PAPER PID {pid}: Windows error {error}")
        return exit_code.value == _STILL_ACTIVE
    finally:
        close_handle(handle)


class PaperProcessLock:
    """Own an atomic lock file while its recorded process remains active."""

    def __init__(self, path: str | Path, *, mode: str = "PAPER") -> None:
        self.path = Path(path)
        self.mode = mode
        self.owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": now_ist().isoformat(),
            "mode": self.mode,
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ProcessLockError("Malformed process lock; refusing ownership") from exc
            if current.get("hostname") != socket.gethostname():
                raise ProcessLockError("Process lock belongs to another host")
            pid = current.get("pid")
            if not isinstance(pid, int) or isinstance(pid, bool):
                raise ProcessLockError("Malformed process lock PID")
            if _is_process_alive(pid):
                raise ProcessLockError(f"PAPER process already active: PID {pid}")
            self.path.unlink()
            return self.acquire()

        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.owned = True

    def release(self) -> None:
        if not self.owned:
            return
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current.get("pid") == os.getpid():
                self.path.unlink()
        finally:
            self.owned = False

    def __enter__(self) -> "PaperProcessLock":
        self.acquire()
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()
