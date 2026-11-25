"""One GPU worker of a pool job: claims tasks from configs/journal/aspire_tasks.tsv and runs them.

    CUDA_VISIBLE_DEVICES=0 python -m scripts.journal.aspire.pool_worker

Task line: "<model>\t<pope_set>\t<condition>\t<extra run_readout args>". Tasks run in file order; the file is
re-read before every claim, so tasks can be appended while jobs run. Claims are atomic directories under
$VCD_OUTPUT_ROOT/journal/_claims/ with a heartbeat file touched every 60 s; a claim whose heartbeat is older than
STALE_S (worker killed, e.g. by walltime) is taken over. run_readout is resumable, so a taken-over task continues
where it stopped. A task that fails twice is marked FAILED and skipped. The worker exits when nothing is runnable.
"""
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TASKS = REPO / "configs" / "journal" / "aspire_tasks.tsv"
OUT = Path(os.environ["VCD_OUTPUT_ROOT"]) / "journal"
CLAIMS = OUT / "_claims"
STALE_S = 20 * 60
ME = f"{socket.gethostname()}:{os.environ.get('PBS_JOBID', '?')}:gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}"


def log(msg):
    print(f"{time.strftime('%F %T')} [{ME}] {msg}", flush=True)


def tasks():
    out = []
    for line in TASKS.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            model, pope_set, cond, *extra = line.split("\t")
            out.append((model, pope_set, cond, extra[0] if extra else ""))
    return out


def task_id(t):
    model, pope_set, cond, extra = t
    view = extra.split("--views", 1)[1].split()[0] if "--views" in extra else "orig"
    tid = f"{pope_set}__{model}__{cond}" + ("" if view == "orig" else f"__{view}")
    sh = shard(extra)
    return tid + (f"__shard{sh[0]}of{sh[1]}" if sh else "")


def shard(extra):
    a = extra.split()
    if "--nshards" in a and int(a[a.index("--nshards") + 1]) > 1:
        return int(a[a.index("--shard") + 1]), int(a[a.index("--nshards") + 1])
    return None


def run_dir(t):
    model, pope_set, cond, extra = t
    view = extra.split("--views", 1)[1].split()[0] if "--views" in extra else "orig"
    d = OUT / pope_set / model / (cond + ("" if view == "orig" else f"__{view}"))
    sh = shard(extra)
    return d / f"shard{sh[0]}of{sh[1]}" if sh else d


def claim(tid) -> Path | None:
    c = CLAIMS / tid
    if (c / "FAILED2").exists():
        return None
    try:
        c.mkdir(parents=True)
    except FileExistsError:
        hb = c / "hb"
        age = time.time() - (hb.stat().st_mtime if hb.exists() else c.stat().st_mtime)
        if age < STALE_S:
            return None
        stale = CLAIMS / f".stale_{tid}_{uuid.uuid4().hex[:6]}"
        try:
            c.rename(stale)          # atomic: only one worker wins the takeover
        except OSError:
            return None
        for f in ("FAILED1",):       # carry the failure count over
            if (stale / f).exists():
                (CLAIMS / f".carry_{tid}").touch()
        try:
            c.mkdir()
        except FileExistsError:
            return None
        if (CLAIMS / f".carry_{tid}").exists():
            (c / "FAILED1").touch()
            (CLAIMS / f".carry_{tid}").unlink(missing_ok=True)
        log(f"took over stale claim {tid}")
    (c / "owner").write_text(ME + "\n")
    (c / "hb").touch()
    return c


def heartbeat(c, stop):
    while not stop.wait(60):
        try:
            (c / "hb").touch()
        except OSError:
            pass


def main():
    CLAIMS.mkdir(parents=True, exist_ok=True)
    while True:
        ran = False
        for t in tasks():
            if (run_dir(t) / "DONE").exists():
                continue
            tid = task_id(t)
            c = claim(tid)
            if c is None:
                continue
            model, pope_set, cond, extra = t
            log(f"START {tid}")
            stop = threading.Event()
            threading.Thread(target=heartbeat, args=(c, stop), daemon=True).start()
            cmd = [sys.executable, "-m", "scripts.journal.run_readout", "--model", model, "--conditions", cond,
                   *extra.split()]
            code = subprocess.call(cmd, cwd=REPO, env={**os.environ, "POPE_SET": pope_set})
            stop.set()
            if code == 0 and (run_dir(t) / "DONE").exists():
                log(f"DONE {tid}")
                for f in c.iterdir():
                    f.unlink()
                c.rmdir()
            else:
                n = 2 if (c / "FAILED1").exists() else 1
                (c / f"FAILED{n}").touch()
                log(f"FAIL {tid} exit {code} (failure {n})")
                if n == 1:   # release for a retry
                    (c / "hb").touch()
                    os.utime(c / "hb", (0, 0))
            ran = True
            break            # re-read the task file after every task
        if not ran:
            log("nothing runnable; exiting")
            return


if __name__ == "__main__":
    main()
