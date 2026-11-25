"""GPU-aware job queue: runs resumable jobs one at a time, yielding the GPU to other people's big jobs.

    python -m scripts.journal.gpu_scheduler                 # usually as a systemd user unit (vcd-sched)

Queue: configs/journal/queue.tsv, one job per line "<name>\t<shell command>" (run from the repo root;
'#' lines ignored). The file is re-read every poll, so jobs can be appended while it runs.
Every 15 s it samples the GPU load of processes that are NOT ours (nvidia-smi compute apps + pmon):
  busy  = others' GPU memory > OTHER_MEM_MB or others' SM utilisation > OTHER_SM_PCT
  pause = busy in >= 3 of the last 4 samples while our job runs: SIGTERM to the job's process group
          (SIGKILL after 60 s). Jobs are resumable per question, so nothing is lost but the model load.
  resume once not busy for RESUME_QUIET_S and at least NEED_FREE_MB of GPU memory is free.
Manual control: create logs/journal/scheduler/PAUSE to pause, delete it to resume.
State (done/failed jobs): logs/journal/scheduler/state.json; one-line events: events.log; job output: <name>.log.
Jobs run strictly in queue order. Exit 0 marks a job done; a crash is retried up to 3 times (60 s apart),
then marked failed, which halts the queue (so a failed smoke-test gate stops the big runs behind it).
"""
import json
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "configs" / "journal" / "queue.tsv"
DIR = REPO / "logs" / "journal" / "scheduler"
STATE, EVENTS, PAUSE_FILE = DIR / "state.json", DIR / "events.log", DIR / "PAUSE"

POLL_S = 15
OTHER_MEM_MB = int(os.environ.get("OTHER_MEM_MB", 6000))
OTHER_SM_PCT = int(os.environ.get("OTHER_SM_PCT", 60))
RESUME_QUIET_S = int(os.environ.get("RESUME_QUIET_S", 300))
NEED_FREE_MB = int(os.environ.get("NEED_FREE_MB", 16000))
MAX_TRIES = 3


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(EVENTS, "a") as f:
        f.write(line + "\n")


def sh(cmd) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except subprocess.TimeoutExpired:
        return ""


def pgroup_pids(pgid) -> set[int]:
    pids = set()
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                stat = (p / "stat").read_text()
                fields = stat[stat.rindex(")") + 2:].split()
                if int(fields[2]) == pgid:       # field 5 (pgrp), after pid/comm/state
                    pids.add(int(p.name))
            except (OSError, ValueError, IndexError):
                pass
    return pids


def gpu_sample(ours: set[int]):
    """(others' compute memory MiB, others' SM %, free memory MiB)."""
    mem = 0
    for line in sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits").splitlines():
        try:
            pid, m = (int(x) for x in line.split(","))
        except ValueError:
            continue
        if pid not in ours:
            mem += m
    sm = 0
    for line in sh("nvidia-smi pmon -c 1 -s u").splitlines():
        f = line.split()
        if line.startswith("#") or len(f) < 4 or not f[1].isdigit():
            continue
        if int(f[1]) not in ours and f[3].isdigit():
            sm += int(f[3])
    try:
        free = int(sh("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits").split()[0])
    except (IndexError, ValueError):
        free = 0
    return mem, sm, free


def read_queue():
    jobs = []
    if QUEUE.exists():
        for line in QUEUE.read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                name, cmd = line.split("\t", 1)
                jobs.append((name.strip(), cmd.strip()))
    return jobs


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(s):
    STATE.write_text(json.dumps(s, indent=1))


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
    except ProcessLookupError:
        pass


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    log(f"scheduler start (pause if others > {OTHER_MEM_MB} MiB or > {OTHER_SM_PCT}% SM; "
        f"resume after {RESUME_QUIET_S}s quiet with >= {NEED_FREE_MB} MiB free)")
    proc, job, window, quiet_since, last_status = None, None, deque(maxlen=4), None, 0
    while True:
        state = load_state()
        ours = pgroup_pids(proc.pid) if proc else set()
        mem, sm, free = gpu_sample(ours)
        busy = mem > OTHER_MEM_MB or sm > OTHER_SM_PCT or PAUSE_FILE.exists()
        window.append(busy)
        quiet_since = None if busy else (quiet_since or time.time())

        if proc is not None:
            code = proc.poll()
            if code is not None:
                name = job[0]
                if code == 0:
                    state[name] = "done"
                    log(f"DONE {name}")
                else:
                    tries = int(str(state.get(name, "tries:0")).split(":")[1]) + 1 if str(
                        state.get(name, "")).startswith("tries") else 1
                    state[name] = f"tries:{tries}" if tries < MAX_TRIES else "failed"
                    log(f"FAIL {name} exit {code} (attempt {tries}/{MAX_TRIES}); see {name}.log")
                    if tries < MAX_TRIES:
                        time.sleep(60)
                save_state(state)
                proc, job = None, None
            elif sum(window) >= 3:
                log(f"PAUSE {job[0]}: others {mem} MiB, {sm}% SM{' (PAUSE file)' if PAUSE_FILE.exists() else ''}")
                stop(proc)
                proc, job, quiet_since = None, None, None

        if proc is None:
            # strictly sequential: the first job that is not done runs next; a failed one halts the queue
            pending = [j for j in read_queue() if state.get(j[0]) != "done"][:1]
            if pending and state.get(pending[0][0]) == "failed":
                if time.time() - last_status > 3600:
                    log(f"HALT: {pending[0][0]} failed; fix it and delete its entry from state.json to continue")
                    last_status = time.time()
            elif not pending:
                if time.time() - last_status > 3600:
                    log("idle: queue empty")
                    last_status = time.time()
            elif quiet_since and time.time() - quiet_since >= RESUME_QUIET_S and free >= NEED_FREE_MB:
                job = pending[0]
                out = open(DIR / f"{job[0]}.log", "a")
                out.write(f"\n===== start {time.strftime('%F %T')} =====\n")
                out.flush()
                proc = subprocess.Popen(["bash", "-c", job[1]], cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                window.clear()
                log(f"START {job[0]} (others {mem} MiB, {sm}% SM, free {free} MiB)")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
