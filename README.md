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
  <img src="https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/main/assets/demo.gif" width="860" alt="slurmwatch live TUI: per-process CPU / memory / GPU bars, a JOB provenance card, and a wall-clock time-budget bar, with the alarm strip lighting up as memory climbs toward the OOM guard.">
</p>

**Did your job actually use the GPUs you asked for?** slurmwatch shows the real per-process numbers live in your terminal — and speaks up only when something needs you.

## Install

```bash
pip install slurmwatch      # or: uv tool install slurmwatch / pipx install slurmwatch
```

Python 3.10+ on Linux (cgroup v1/v2). GPU monitoring auto-activates on NVIDIA nodes, and is skipped on CPU-only ones.

## Usage

```bash
slurmwatch 12345    # watch a specific job
slurmwatch          # or auto-discover your running job
sw 12345            # "sw" is a short alias
slurmwatch --help   # everything else
```

**Keys** — `c`/`m`/`g` drill into CPU / memory / GPU · **type a node number** (or `←`/`→`) to switch node · `p` expand a truncated path · `q` back/quit.

## What you get

- **An alarm strip that stays quiet.** It shows a line only when something is actionable — `MEMORY 91% of limit`, `1 OF 2 GPUS IDLE`. A healthy job gets none.
- **Bars that state facts, not verdicts.** CPU, memory, and each GPU's compute + VRAM, with the live figures and a recent range. Colour is identity — **CPU cyan · memory rose · GPU violet/teal** — its length is how full, and *you* judge whether that's good. The classic idle-but-holding-VRAM case is obvious at a glance.
- **Per-process honesty.** NVML and cgroups count only *your* PIDs, so a neighbour on a shared node never inflates your numbers; memory is the real working set, not cache-padded.
- **Context that's usually a `scontrol` away.** A JOB card with the launch command, workdir, and log paths; a bottom bar tracking elapsed vs the wall-clock limit and when you'll hit it.
- **Every node, one process.** Type a node's number to jump straight to node 199 of a 200-node job.

Point it at a **pending** job and, instead of an error, you get *why* it's waiting (the Slurm reason in plain English), *when* the scheduler thinks it'll start and where you sit in line, and *where* it could run right now — with the exact `scontrol update` to requeue into a partition that fits.

Run it from a login node or on the node itself — it attaches either way, and falls back to an `sstat` summary when it can't.
