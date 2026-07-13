<h1 align="center">slurmwatch</h1>

<p align="center">
  <strong>Live per-process CPU / memory / GPU telemetry for a running Slurm job — the facts, so you can judge.</strong>
</p>

<p align="center">
  <a href="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml"><img src="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/slurmwatch/"><img src="https://img.shields.io/pypi/v/slurmwatch.svg?cache=bust" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/lint-ruff-261230.svg" alt="Ruff"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/main/assets/demo.gif" width="860" alt="slurmwatch live TUI dashboard showing per-process CPU, memory, and GPU telemetry for a Slurm job.">
</p>

## Install

```bash
pip install slurmwatch      # or: uv tool install slurmwatch / pipx install slurmwatch
```

Python 3.10+ on Linux (cgroup v1/v2). GPU monitoring (`pynvml`) auto-activates on NVIDIA nodes and is skipped on CPU-only ones.

## Usage

```bash
slurmwatch                       # auto-discover and attach to your running job
slurmwatch 12345                 # a specific job (array 12345_3, het 12345+1)
sw 12345                         # "sw" is a short alias
slurmwatch --demo                # live TUI, no Slurm needed
slurmwatch --demo pending        # preview the pending-job view offline
slurmwatch 12345 --once --json   # one machine-readable snapshot, then exit
slurmwatch 12345 --log run.jsonl # headless logging (JSONL or CSV)
```

Point it at a **pending** job and, instead of an error, you get why it's waiting, when it should start, and where it could run — no flags needed.

**Keys** — `c`/`m`/`g` open a full-screen CPU/memory/GPU drill-in (in the GPU view `↑`/`↓` pick a device) · **type a node number** (or `←`/`→`) to switch node · `p` reveal a truncated path · `↑`/`↓` `PgUp`/`PgDn` scroll · `q` back/quit.

## Notes

- From a login node it attaches to the compute node via `srun --overlap`, so the view runs inside your allocation. The attach is bounded and always opens the dashboard (CPU/memory/processes) — even when the GPU can't be read.
- **Live GPU util needs the GPU to be reachable from a monitor step.** If your batch script launches the GPU program *directly* (`python train.py`), the GPU sits in the `.batch` step and slurmwatch reads it. If it launches via an **inner `srun`** (`srun python train.py`), that step *locks* the GPU exclusively (Slurm won't share a GPU across steps and blocks device access via cgroups), so a separate monitor step can't read GPU util — the dashboard shows everything else live and says so. To get live GPU on such a job, run the program without the inner `srun` (multi-node distributed training that must use `srun` is the exception — there's no way to read its GPU after the fact).
- Can't attach at all? You get an `sstat` summary — peak memory, CPU time, allocation — but no live GPU utilization, which Slurm doesn't track per device.
- `SLURMWATCH_NO_HOP=1` forces the summary · `--ascii` for a non-UTF-8 terminal · `SLURMWATCH_MOUSE=1` enables the wheel (off by default so text selection works).
- Everything else: `slurmwatch --help` and the `SLURMWATCH_*` env vars.

## Features

- **Facts, not verdicts** — labelled bars (`usage` · `used` · `compute` · `vram`), each with its recent 60-second range and a health dot (`●`/`▲`/`✖`). An alarm strip surfaces only what needs action (`MEMORY 91% of limit`, `1 OF 2 GPUS IDLE`).
- **Drill in** — `c`/`m`/`g` open a focused full-screen view of one resource: the live figure in large digits plus a 60-second area chart of its recent history. In the GPU view, arrow through devices and the chart follows the one you pick.
- **Per-process** — NVML and cgroups count only *your* PIDs, so a neighbour on a shared node never inflates your numbers.
- **Honest memory** — working set (RSS minus reclaimable cache), against a configurable OOM guard.
- **Multi-node** — one process, every node: type a node's number (or step with `◂ ▸`) to switch which node the dashboard shows — jump straight to node 199 of a 200-node job.
- **Pending jobs** — point it at a queued job and instead of an error you get *why* it's waiting (the Slurm reason, in plain English), *when* the scheduler estimates it'll start, and *where* it could run right now — a cluster-wide view of free capacity that flags any partition your request would fit into, with the exact `scontrol update` to requeue there.
- **Runs anywhere** — full live telemetry on the node; falls back to Slurm accounting (`sstat`) when it can't attach.
- **Zero config** — auto-discovers the job, cgroup v1/v2, GPUs, and where it's running.
