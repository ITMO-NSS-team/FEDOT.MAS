from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

GUARD = [sys.executable, "-m", "mcp_browser_usage._guard"]

# Stands in for browser-use: starts a long-lived descendant (Chrome), records
# its pid, then idles.
SPAWNER = """
import subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
with open(sys.argv[1], "w") as f:
    f.write(f"{grandchild.pid}")
time.sleep(60)
"""

# Stands in for the proxy: starts the guard with its own pid, then either
# idles or exits as soon as the grandchild is up, depending on argv[1].  The
# last argument is the spawner's pid file.
PROXY = """
import os, subprocess, sys, time
subprocess.Popen(
    [sys.executable, "-m", "mcp_browser_usage._guard", str(os.getpid()), *sys.argv[2:]]
)
if sys.argv[1] == "idle":
    time.sleep(60)
while not (os.path.exists(sys.argv[-1]) and open(sys.argv[-1]).read()):
    time.sleep(0.05)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _read_pid(path, timeout=10.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError("spawner never reported its grandchild")


def _wait_gone(pid: int, timeout=10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.1)
    return False


def _spawn_proxy(mode, pid_file):
    return subprocess.Popen(
        [sys.executable, "-c", PROXY, mode, sys.executable, "-c", SPAWNER, str(pid_file)]
    )


class TestGuard:
    def test_exit_code_is_passed_through(self):
        done = subprocess.run(
            [*GUARD, str(os.getpid()), sys.executable, "-c", "import sys; sys.exit(3)"],
            timeout=30,
        )
        assert done.returncode == 3

    def test_missing_command_is_a_usage_error(self):
        done = subprocess.run(
            [*GUARD, str(os.getpid())], capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 2
        assert "usage" in done.stderr

    def test_non_numeric_proxy_pid_is_a_usage_error(self):
        done = subprocess.run(
            [*GUARD, "uvx", "browser-use"], capture_output=True, text=True, timeout=30
        )
        assert done.returncode == 2

    def test_proxy_death_takes_the_descendants_down(self, tmp_path):
        pid_file = tmp_path / "pid"
        proxy = _spawn_proxy("idle", pid_file)
        try:
            grandchild = _read_pid(pid_file)
            assert _alive(grandchild)
            proxy.send_signal(signal.SIGKILL)
            proxy.wait(timeout=10)
            assert _wait_gone(grandchild)
        finally:
            if proxy.poll() is None:
                proxy.kill()

    def test_an_unreaped_proxy_counts_as_dead(self, tmp_path):
        """A parent stuck draining stderr never reaps the proxy it killed."""
        pid_file = tmp_path / "pid"
        # Not waited on until the end, so it stays a zombie meanwhile.
        proxy = _spawn_proxy("exit", pid_file)
        try:
            grandchild = _read_pid(pid_file)
            assert _wait_gone(grandchild)
        finally:
            proxy.wait(timeout=10)

    def test_proxy_already_gone_takes_the_command_down(self):
        # getppid() of an orphan is 1; the watcher must not wait on init.
        done = subprocess.run(
            [*GUARD, "1", sys.executable, "-c", "import time; time.sleep(30)"],
            timeout=20,
        )
        assert done.returncode != 0

    def test_command_exit_takes_its_descendants_down(self, tmp_path):
        pid_file = tmp_path / "pid"
        guard = subprocess.Popen(
            [*GUARD, str(os.getpid()), sys.executable, "-c", SPAWNER, str(pid_file)]
        )
        try:
            grandchild = _read_pid(pid_file)
            guard.send_signal(signal.SIGTERM)
            guard.wait(timeout=10)
            assert _wait_gone(grandchild)
        finally:
            if guard.poll() is None:
                guard.kill()
