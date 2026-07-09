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

Run `slurmwatch <jobid>` from a login node — it attaches to your job's compute node for you and shows a live dashboard. No manual `srun`, no flags to memorize.

## Features

- **Facts, not verdicts** — labelled bars (`usage` · `used` · `compute` · `vram`), each with its recent 60-second range and a health dot (`●`/`▲`/`✖`). An alarm strip surfaces only what needs action (`MEMORY 91% of limit`, `1 OF 2 GPUS IDLE`).
- **Per-process** — NVML and cgroups count only *your* PIDs, so a neighbour on a shared node never inflates your numbers.
- **Honest memory** — working set (RSS minus reclaimable cache), against a configurable OOM guard.
- **Multi-node** — one process, every node: type a node's number (or step with `◂ ▸`) to switch which node the dashboard shows — jump straight to node 199 of a 200-node job.
- **Runs anywhere** — full live telemetry on the node; falls back to Slurm accounting (`sstat`) when it can't attach.
- **Zero config** — auto-discovers the job, cgroup v1/v2, GPUs, and where it's running.

## Install

```bash
pip install slurmwatch      # or: uv tool install slurmwatch / pipx install slurmwatch
```

Python 3.10+ on Linux (cgroup v1 or v2). One install fits a mixed cluster — NVIDIA GPU monitoring (via `pynvml`) auto-activates on GPU nodes and is skipped on CPU-only ones.

## Usage

```bash
slurmwatch                       # auto-discover and attach to your running job
slurmwatch 12345                 # a specific job (array 12345_3, het 12345+1)
sw 12345                         # "sw" is a short alias
slurmwatch --demo                # try the live TUI now — no Slurm needed
slurmwatch 12345 --once --json   # one machine-readable snapshot, then exit
slurmwatch 12345 --log run.jsonl # headless logging (JSONL or CSV)
```

**Keys** — `c` / `m` / `g`: CPU / memory / GPU detail view · **type a node number** (or `◂ ▸`): switch node on a multi-node job · `p`: reveal a truncated command/workdir path · `↑ ↓` `PgUp` `PgDn`: scroll · `q`: quit.

From a login node, slurmwatch attaches to the compute node via `srun --overlap` (so the view runs inside your allocation). Can't attach? It shows an `sstat` summary — peak memory, CPU time, allocation — but not live GPU utilization, which Slurm doesn't track per device. `SLURMWATCH_NO_HOP=1` forces the summary; `--ascii` suits a non-UTF-8 terminal; `SLURMWATCH_MOUSE=1` enables the wheel (mouse capture is off by default so text selection works). See `slurmwatch --help` and the `SLURMWATCH_*` env vars for the rest.

## Library

```python
import asyncio
from slurmwatch import TelemetryCollector, resolve_job_context

async def sample(job_id):
    collector = TelemetryCollector(resolve_job_context(job_id))
    await collector.start()
    try:
        print((await collector.next_snapshot()).to_json())
    finally:
        await collector.stop()

asyncio.run(sample("12345"))
```

## Limitations

- NVIDIA only (no AMD/ROCm).
- One node on screen at a time — switch by typing its number (or `◂ ▸`); non-local nodes refresh a few seconds slower. No cross-node aggregate.
- Live GPU utilization and working-set memory need to run on the job's node.

## License

MIT
