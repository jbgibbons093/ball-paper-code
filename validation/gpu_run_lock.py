#!/usr/bin/env python
"""Run one command while holding the repository-wide BALL GPU lock.

The lock blocks indefinitely and is released by the operating system if this
wrapper exits unexpectedly. This prevents two long BALL training pipelines from
silently sharing the single CUDA device.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


class GpuRunLock:
    def __init__(self, name: str = "BALL_GPU_PIPELINE_SINGLE_DEVICE") -> None:
        self.name = name
        self.handle = None
        self.file_handle = None

    def __enter__(self):
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint32
            self._kernel32 = kernel32
            self.handle = kernel32.CreateMutexW(None, False, f"Global\\{self.name}")
            if not self.handle:
                raise OSError(ctypes.get_last_error(), "CreateMutexW failed")
            wait_result = kernel32.WaitForSingleObject(self.handle, 0xFFFFFFFF)
            if wait_result not in (0x00000000, 0x00000080):
                raise OSError(ctypes.get_last_error(), f"WaitForSingleObject failed: {wait_result}")
        else:
            import fcntl

            lock_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"{self.name}.lock"
            self.file_handle = lock_path.open("a+b")
            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if os.name == "nt" and self.handle:
            self._kernel32.ReleaseMutex(self.handle)
            self._kernel32.CloseHandle(self.handle)
            self.handle = None
        elif self.file_handle is not None:
            import fcntl

            fcntl.flock(self.file_handle.fileno(), fcntl.LOCK_UN)
            self.file_handle.close()
            self.file_handle = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Serialize BALL commands on one GPU")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = list(arguments.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    print(f"waiting for exclusive BALL GPU access: {' '.join(command)}", flush=True)
    with GpuRunLock():
        print("exclusive BALL GPU access acquired", flush=True)
        completed = subprocess.run(command, check=False)
    print(f"exclusive BALL GPU access released; exit code {completed.returncode}", flush=True)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
