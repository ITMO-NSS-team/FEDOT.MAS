"""Run a command so that its whole process tree dies with the proxy.

Usage: ``python -m mcp_browser_usage._guard PROXY_PID COMMAND [ARGS...]``

browser-use launches Chrome in its own process group, and nothing ends that
group when the proxy goes: browser-use does not exit on a closed stdin while a
browser session is open, and a proxy that dies without closing (or whose client
gives up first) kills nothing.  browser-use and Chrome then outlive the run,
keep the inherited stderr open, and a run piped through ``tee`` never sees end
of file.

This process execs the command, so the command keeps this pid and group, and a
kill aimed at the guard's group reaches browser-use and Chrome.  For every case
where no such kill comes, a watcher in a session of its own -- beyond the reach
of any group kill, and with no inherited stdio to hold open -- terminates the
group once the proxy or the command's own process is gone.

The proxy passes its pid rather than the guard reading ``getppid()``: a proxy
killed while the guard's interpreter starts would already have handed the
guard to init.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

POLL_SECONDS = 0.5
GRACE_SECONDS = 3.0
USAGE = "usage: python -m mcp_browser_usage._guard PROXY_PID COMMAND [ARGS...]"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--watch"] and len(args) == 3:
        return _watch(proxy=int(args[1]), leader=int(args[2]))
    if len(args) < 2 or not args[0].isdigit():
        print(USAGE, file=sys.stderr)
        return 2
    proxy, command = args[0], args[1:]

    # A session leader already leads its group and may not call setpgid.
    if os.getpgrp() != os.getpid():
        os.setpgid(0, 0)
    subprocess.Popen(
        [sys.executable, "-m", "mcp_browser_usage._guard", "--watch"]
        + [proxy, str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.execvp(command[0], command)


def _watch(proxy: int, leader: int) -> int:
    # The leader's pid is its group's id; exec keeps the pid.  Pid 1 means the
    # proxy was gone before it could be recorded.
    while proxy != 1 and _alive(proxy) and _alive(leader) and _group_alive(leader):
        time.sleep(POLL_SECONDS)
    _terminate_group(leader)
    return 0


def _terminate_group(pgid: int) -> None:
    if not _signal_group(pgid, signal.SIGTERM):
        return
    deadline = time.monotonic() + GRACE_SECONDS
    while time.monotonic() < deadline:
        if not _group_alive(pgid):
            return
        time.sleep(0.1)
    _signal_group(pgid, signal.SIGKILL)


def _signal_group(pgid: int, signum: int) -> bool:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def _group_alive(pgid: int) -> bool:
    return _signal_group(pgid, 0)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    # kill(0) succeeds on a zombie, and a parent busy draining the stderr that
    # Chrome still holds may never reap one.
    return not _is_zombie(pid)


def _is_zombie(pid: int) -> bool:
    if os.path.isdir("/proc/self"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] == "Z"
        except (OSError, IndexError):
            return False
    try:
        stat = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except OSError:
        return False
    return stat.strip().startswith("Z")


if __name__ == "__main__":
    sys.exit(main())
