# GPU sandbox (lum.id) — how to connect and work

A 24-hour single-GPU sandbox, first provisioned 2026-09-27 ~15:30 local. Complements the
NUS cluster (`CLUSTER.md`): no queue, no wall-time cap, but one GPU and a hard expiry.

Official guide: <https://lum.id/docs/sandboxes.md> (raw markdown; the Studio page at
lum.id/studio/docs/sandboxes renders it client-side). Sandboxes are created, deleted and
restarted in **Studio → Research Fleet → Sandboxes**. Per user per site the limits are
2 sandboxes, 24 h TTL, and 1 GPU at `home`.

## Connect

```bash
chmod 600 .ssh                                   # the key is the file ./.ssh in the repo root
ssh -i .ssh -p 31223 gw@lum.id                   # lands as root in /workspace
```

Optional `~/.ssh/config` alias, then `ssh sbx`:

```
Host sbx
    HostName lum.id
    Port 31223
    User gw
    IdentityFile ~/fyp/.ssh
```

**`./.ssh` is a private key. It is listed in `.gitignore` — never commit it.**

## What is there (measured 2026-09-27)

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4000 Blackwell SFF, 24 GB, compute capability 12.0 (sm_120), driver 595.91, CUDA 13.2 |
| CPU / RAM | 32 cores / 29 GB (+1 GB swap) |
| OS | Ubuntu 24.04, Python 3.12.3, `uv` and `pip3` present; no `nvcc`, `curl`, `rsync`, `tmux` |
| **real limits** | **cgroup: 8 GB RAM** (`/sys/fs/cgroup/memory.max` = 8589934592) and **2 CPUs** (`cpu.max` = `200000 100000`). `free` shows 29 GB and `nproc` shows 32, and **both are the host's figures, not yours** |
| user | `root` (so `sudo` is neither needed nor installed; `apt-get install` works directly) |
| network | outbound HTTPS works (PyPI, Hugging Face, download.pytorch.org, GitHub) |

## The two traps

1. **Only `/home/riwk2015` persists.** It is an 84 TB NFS mount. Everything else —
   including `$HOME=/root`, `/workspace`, and the 938 GB overlay on `/` — is wiped.
   Put the repo, venv, and every cache there:
   ```bash
   export HF_HOME=/home/riwk2015/.cache/hf UV_CACHE_DIR=/home/riwk2015/.cache/uv \
          PIP_CACHE_DIR=/home/riwk2015/.cache/pip TORCH_HOME=/home/riwk2015/.cache/torch
   ```
   A pre-existing `/home/riwk2015/.venv` (Python 3.12) is there. For heavy scratch I/O,
   the overlay `/` is local disk and faster than NFS, but copy results back before expiry.
2. **`scp` fails** (`scp: Connection closed`). The guide says `scp -O` works, but only if
   the image has an `scp` binary, and this one does not (`scp: not found`); nor `rsync`.
   `apt-get install -y openssh-client rsync` would fix that. Until then, pipe through
   `ssh` (~15 MB/s each way):
   ```bash
   # up
   tar czf - hrt/artifacts/panel.npz | ssh -i .ssh -p 31223 gw@lum.id 'tar xzf - -C /home/riwk2015/fyp'
   # down
   ssh -i .ssh -p 31223 gw@lum.id 'tar czf - -C /home/riwk2015/fyp hrt/artifacts/runs' | tar xzf -
   ```
   Verify big transfers with `md5sum` on both sides. Large public data (Polymarket-v1,
   16.8 GB) is faster to re-download on the sandbox from Hugging Face than to upload.

## Gateway commands and the one-sandbox rule

The gateway accepts `sbx ls`, `sbx enter NAME`, `sbx logs NAME` and `sbx data`. Anything
else runs in your **first sandbox alphabetically**, and there is no non-interactive way
to target one by name. **Keep one sandbox per site**, or a one-shot `ssh … '<cmd>'` can
land in the wrong box.

The FinData, LQT and Lumid Data stores are injected as `FINDATA_URL`, `LQT_DATA_URL` and
`LUMID_DATA_URL` (FinData is anon-read; query server-side rather than copying). `/datasets`
is a per-site read-only file mount (only `symbols` here). The guide mentions a node-local
`/huggingface` weight cache, but **this box has none**, so models download to `HF_HOME`.
The guide's **Save** button commits the container to `harbor.lum.id/sbx-<you>/…` (20 GB
cap), which is how to keep `/opt/venv` across restarts.

## It can vanish before you expect it to

After the Studio restart at ~16:25 UTC on 2026-09-27, the sandbox was **gone by 06:18 UTC on
09-28**, about 14 h later, not 24. The gateway then answers
`you have no sandbox yet — create one in Studio → Compute`. The TTL evidently counts from
the original creation, not from a restart. Assume the expiry is the *earliest* plausible
time, and pull results continuously (`scripts/pull_probe.sh` reconciles by name and size
every hour; a timestamp marker had silently dropped files when one tar stream broke). The
home directory survives on the NAS per the guide, so a re-created sandbox with the same
name should see `/home/riwk2015` again.

## Long jobs

There is no `tmux` (install it with `apt-get install -y tmux` if wanted). A detached
process **survives the ssh session closing** (verified):

```bash
ssh -i .ssh -p 31223 gw@lum.id 'cd /home/riwk2015/fyp && setsid nohup ./job.sh > logs/job.log 2>&1 < /dev/null &'
```

Write logs and results under `/home/riwk2015`, keep jobs resumable (the sweep scripts
already skip runs whose JSON exists), and pull results down well before the 24 h expiry.

## Trap 3 — running out of memory kills the whole pod, not just the process

On 2026-09-27, 12 concurrent `train.py --agent hrt` calibration runs exceeded memory.
The kernel did not just kill one worker: the **pod went to phase `Failed`** and every
ssh after that returned
`error: cannot exec into a container in a completed pod; current phase is Failed`.
It did not restart by itself within 5 minutes, and nothing on our side can restart it.
Whatever was on local disk (`/opt/venv`, `/root/.cache`) went with it.

The cause: the pod's cgroup limit is **8 GB**, not the 29 GB that `free` shows. So run
at most **2 HRT workers**, which the 2-CPU quota also dictates, and read the cgroup's own
counter (`cat /sys/fs/cgroup/memory.current`), never `free`. A restart from Studio brings
the pod back with `/home/riwk2015` intact. Each worker loads
the 440 MB `panel.npz` (larger once decompressed) as well as its 2 GB replay buffer.

## Python environment

Installing into a venv on the NFS home stalls: 10+ minutes for torch without finishing,
because it writes thousands of small files. Install on local disk instead (13 s):

```bash
bash -c 'export UV_CACHE_DIR=/root/.cache/uv
uv venv -q --python 3.12 /opt/venv
uv pip install -q --python /opt/venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install -q --python /opt/venv/bin/python numpy pandas scipy pyarrow duckdb scikit-learn transformers accelerate'
```

torch 2.11.0+cu128 sees the GPU as capability (12, 0). `/opt/venv` does not survive a pod
reset, so rebuild it after one. The login shell is `sh` (no `time` builtin), so wrap
anything non-trivial in `bash -c`. Never `pkill -f` with a pattern that also appears in the
command you are running: it kills its own shell. Bracket one letter
(`pkill -f "[t]rain.py"`), and never put the kill and a relaunch of the same
script in one command line: the relaunch text matches too. Or kill by PID.

## Sizing for the RL sweep

Measured on this box: one HRT run alone does 12,288 steps in 20 s (~600 steps/s, with
`--probe` on), so a full 500k-step run is ~15 min without contention.


From `CLUSTER.md` §6: ~2.1 GB host RAM and 2 CPU threads per worker, GPU memory trivial.
Under the 8 GB / 2-CPU cgroup that means **2 concurrent workers**.
The workload is kernel-launch-bound, so calibrate steps/s with the 12,288-step `_cal`
run before sizing a sweep.
