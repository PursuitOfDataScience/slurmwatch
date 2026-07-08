<h1 align="center">slurmwatch</h1>

<p align="center">
  <strong>Live, per-process CPU / memory / GPU telemetry for a running Slurm job — the facts, so you can judge.</strong>
</p>

<p align="center">
  <a href="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml"><img src="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/slurmwatch/"><img src="https://img.shields.io/pypi/v/slurmwatch.svg?cache=bust" alt="PyPI"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/lint-ruff-261230.svg" alt="Ruff"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/main/assets/demo.gif" width="860" alt="slurmwatch live TUI dashboard: per-process CPU, memory, and GPU telemetry for a Slurm job. Labelled bars with per-row recent-range tags, a JOB provenance card, and a docked job-info bar; the memory row's dot and the alarm strip light up amber then red as working-set memory climbs toward the limit, while an idle GPU is flagged (1 of 2).">
</p>

## Features

- **Facts-first dashboard** — report the numbers, let *you* judge. Labelled bars (`compute` / `vram` / `usage` / `used`) each carry their recent 60-second range, and a coloured health dot (`●` healthy / `▲` warning / `✖` critical) is the only status channel — no "underused / idle" verdict words. An alarm strip surfaces only things that need action, as facts (`MEMORY 91% of limit`, `1 OF 2 GPUS IDLE`). A **JOB card** shows the run's provenance (account · QOS · state, command, workdir, queue wait), and a docked bottom bar tracks the live time budget.
- **Per-process GPU attribution** — NVML sees only *your* PIDs, so a neighbor's job never inflates your numbers.
- **Honest memory** — working set (RSS minus reclaimable cache) with a configurable OOM guard.
- **Works anywhere** — full live telemetry on the node; auto-falls back to Slurm accounting (`sstat`) from a login node.
- **Zero config** — `slurmwatch <jobid>` auto-discovers jobs, cgroup v1/v2, and whether it's on the node. No flags to memorize.

## Install

```bash
pip install slurmwatch
```

Requires Python 3.10+ and Linux with cgroup v1 or v2. One install works across a mixed cluster: GPU monitoring (NVIDIA, via `pynvml`) auto-activates on GPU nodes and is silently skipped on CPU-only nodes. Works with `pipx` / `uv` too — e.g. `uv tool install slurmwatch`.

## Usage

```bash
slurmwatch                       # auto-discover and attach to your running job
slurmwatch 12345                 # attach to a job (array: 12345_3, het: 12345+1)
sw 12345                         # "sw" is a short alias for "slurmwatch"
slurmwatch --demo                # try the live TUI right now — no Slurm needed
slurmwatch 12345 --once --json   # one machine-readable snapshot, then exit
slurmwatch 12345 --log run.jsonl # headless logging (JSON Lines or CSV)
```

Run it from anywhere: on a login node, slurmwatch automatically attaches to the job's compute node (via `srun --overlap`) to show the live dashboard — no manual `srun` needed. If it can't attach, it falls back to an `sstat` summary (peak memory + CPU time + allocation); GPU *utilization* isn't available that way, since Slurm tracks GPU count, not per-device util. Set `SLURMWATCH_NO_HOP=1` to skip the hop and always get the summary. (The attached view runs inside the job's allocation, so it counts against the job's resources.)

TUI keys: `c`/`m`/`g` open a CPU / memory / GPU detail view — the memory view breaks down working set vs. cache and the headroom to the OOM line, and the GPU view shows this job's per-device share (`JOB%` / `JOB VRAM`), each over a full-height history graph. On a multi-node job, `[` / `]` switch which node the dashboard shows (the current node is live; others are sampled on demand, marked "sampled Ns ago"). Arrows/`PgUp`/`PgDn` scroll and `q` quits. Mouse capture is off so you can select and copy text normally; set `SLURMWATCH_MOUSE=1` to enable mouse/wheel support instead.

Exit codes: `0` success · `1` runtime failure · `2` bad config. Errors go to stderr, so piped `--once`/`--log` output stays clean.

See `slurmwatch --help` for the full flag list. Behavior is also tunable via `SLURMWATCH_*` environment variables — e.g. `SLURMWATCH_OOM_WARN`, `SLURMWATCH_GPU_IDLE_PCT`, `SLURMWATCH_POLL_INTERVAL` (plus ASCII mode and more).

## Library

```python
import asyncio
from slurmwatch import TelemetryCollector, resolve_job_context

async def sample(job_id: str):
    collector = TelemetryCollector(resolve_job_context(job_id))
    await collector.start()
    try:
        print((await collector.next_snapshot()).to_json())
    finally:
        await collector.stop()

asyncio.run(sample("12345"))
```

## Limitations

- NVIDIA-only GPU support (no AMD/ROCm).
- One node on screen at a time — a multi-node job shows a single node's live data, with `[` / `]` to switch between nodes (non-local nodes are sampled on demand via `srun`, so they refresh a few seconds slower). No cross-node aggregate view.
- Live GPU utilization and working-set memory require running on the job's node.

## License

MIT
