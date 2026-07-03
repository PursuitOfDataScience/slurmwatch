# slurmwatch

Live, process-isolated node-local hardware telemetry for active Slurm jobs.

Monitor CPU, memory, and GPU utilization of running Slurm jobs in real time,
with per-process GPU attribution and allocation-efficiency analysis.

<p align="center">
  <img src="https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/main/assets/demo.gif" width="840" alt="slurmwatch live TUI dashboard: per-process CPU, memory, and GPU telemetry for a Slurm job, with an allocation-efficiency verdict flagging an idle GPU">
</p>

## Requirements

- **Python 3.10+**
- **Slurm compute node** — slurmwatch must run on a node where the job is actively
  running (login nodes and non-compute nodes will not have the job's cgroup filesystem).
- **Linux** with cgroup v1 or v2
- **NVIDIA GPU monitoring** (optional): `pip install "slurmwatch[nvidia]"`

## Installation

```bash
pip install slurmwatch

# With NVIDIA GPU support:
pip install "slurmwatch[nvidia]"

# Or as an isolated tool:
pipx install "slurmwatch[nvidia]"
uv tool install "slurmwatch[nvidia]"
```

## Quick Start

```bash
# Attach to a running job (interactive TUI)
slurmwatch <job_id>

# Auto-discover your running jobs (picker appears if you have several)
slurmwatch

# Try it right now with simulated data — no Slurm or GPUs needed
slurmwatch --demo

# One-shot snapshot to stdout (CSV by default, --json for JSON)
slurmwatch <job_id> --once --json

# Headless logging (JSON Lines or CSV, chosen by extension or --format)
slurmwatch <job_id> --log metrics.jsonl
slurmwatch <job_id> --log metrics.csv
```

## Usage

**Important:** slurmwatch must run on a compute node where the job is executing.
Use `srun --jobid <job_id> --overlap slurmwatch <job_id>` if needed.

### Command-line options

| Option | Description |
|--------|-------------|
| `job_id` | Job to monitor (optional; auto-discovers your running jobs). Array tasks like `12345_3` and het-job components like `12345+1` are resolved to the right cgroup. |
| `--once` | Take a single snapshot, print to stdout, exit |
| `--log FILE` | Run headless, appending telemetry snapshots to FILE |
| `--append` | With `--log`, append to an existing file instead of overwriting |
| `--format {json,csv}` | Output format for `--once` and `--log` (default: `--log` infers from the extension, otherwise JSON; `--once` prints CSV) |
| `--json` | Shorthand for `--format json` |
| `--interval SECONDS` | Polling interval (default: 0.5 TUI, 1.0 headless; must be > 0) |
| `--ascii` | ASCII-only output (no Unicode block glyphs) |
| `--demo` | Simulated demo data — no Slurm needed (equivalent to `SLURMWATCH_MOCK=1`) |
| `--verbose` | Verbose diagnostic logging on stderr |
| `--version` | Print version and exit |

Exit codes: `0` success, `1` runtime failure (job not found, not on the right
node, Slurm query failed), `2` invalid configuration. Errors go to stderr, so
`--once`/`--log` output stays clean for pipelines.

### Interactive TUI

| Key | Action |
|-----|--------|
| `c` | Focus CPU panel |
| `m` | Focus Memory panel |
| `g` | Focus GPU panel |
| `v` | Focus Allocation Verdict |
| `q` / `Escape` | Quit |
| `Up` / `Down` | Scroll |
| `PgUp` / `PgDn` | Page scroll |

With no `job_id`, slurmwatch lists your running jobs; pick one with the arrow
keys and `Enter` (or click).

### Headless Mode

Write telemetry as JSON Lines or CSV:

```bash
slurmwatch 12345 --log metrics.jsonl
slurmwatch 12345 --log metrics.csv

# In a batch script (use --append so requeued jobs don't truncate the log):
slurmwatch $SLURM_JOB_ID --log "${SLURM_JOB_ID}.jsonl" --append &
```

Job-array tasks are supported — `slurmwatch 12345_3` resolves the task's own
raw JobId for cgroup discovery, so the right task is monitored even when
several array elements share a node.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURMWATCH_MOCK=1` | — | Enable demo/simulation mode (no Slurm needed) |
| `SLURMWATCH_POLL_INTERVAL` | `0.5` | TUI polling interval in seconds (min 0.05) |
| `SLURMWATCH_HEADLESS_INTERVAL` | `1.0` | Headless polling interval in seconds (min 0.05) |
| `SLURMWATCH_OOM_WARN` | `0.85` | Memory OOM warning threshold (fraction of limit) |
| `SLURMWATCH_OOM_CRIT` | `0.90` | Memory OOM critical threshold (fraction of limit) |
| `SLURMWATCH_HISTORY_SECONDS` | `60` | Sparkline history length in seconds |
| `SLURMWATCH_CPU_UNDERUSE` | `0.5` | Warn when fewer effective cores than this are used on a multi-core allocation |
| `SLURMWATCH_GPU_IDLE_PCT` | `5.0` | Per-process GPU utilization (%) below which a GPU counts as idle |
| `SLURMWATCH_ASCII` | `0` | Use ASCII-only characters (`1` or `true`) |
| `SLURMWATCH_FORMAT` | — | Default `--log`/`--once` format (`json` or `csv`); explicit `--format` wins |
| `SLURMWATCH_CSV_DIALECT` | `excel` | Python `csv` dialect used for CSV output |

## What It Does

### CPU
- Real-time utilization as a percentage of the CPUs allocated on this node
  (multi-node jobs are scaled to node-local limits)
- Reads the `cpuacct`/`cpu.stat` cgroup accounting when present, and falls back
  to summing `/proc/<pid>/stat` across the job's processes — so CPU is measured
  even on clusters that constrain jobs with `cpuset` only (no per-job `cpuacct`
  cgroup)
- **Effective cores** readout — how many cores are actually being used (1.2 / 16 means
  ~1.2 cores' worth of work on a 16-core allocation)
- Underutilization warnings when effective cores fall below
  `SLURMWATCH_CPU_UNDERUSE` on a multi-core allocation

### Memory
- **Working set** tracking — subtracts reclaimable page cache from `memory.current`
  to show actual job memory usage
- Peak memory tracking (with fallback on kernels < 5.19 that lack `memory.peak`)
- OOM guard with configurable warning/critical thresholds
- Falls back to node physical RAM when cgroup limit is unlimited

### GPU (NVIDIA only, requires `pynvml`)
- **Correct device selection** — allocated GPU indices come from
  `scontrol show job -d` (IDX list), with CUDA_VISIBLE_DEVICES UUID/MIG tokens
  from the job's processes as a complement, so the right physical GPUs are
  monitored even with `ConstrainDevices` or multiple jobs per node
- **Per-process GPU attribution** — reads the job's PIDs from its cgroup
  (including cgroup v2 leaf cgroups) and uses
  `nvmlDeviceGetComputeRunningProcesses` / `nvmlDeviceGetGraphicsRunningProcesses`
  to attribute only this job's GPU memory usage
- **Per-process SM utilization** via `nvmlDeviceGetProcessUtilization`,
  summed across the job's processes per device
- Device-wide utilization, VRAM usage, power, and temperature
- Genuine throttling detection (hardware slowdown, power brake, thermal events)
- "Requested vs used" comparison for GPUs; CPU-only jobs never display other
  users' GPUs

### Allocation Verdict
- Summary panel showing whether CPU, memory, and GPU resources are being used optimally
- Flagged underutilization: idle GPUs, single-core workloads on large allocations,
  negligible memory pressure

## Library Use

The building blocks are importable for your own tooling:

```python
import asyncio
from slurmwatch import TelemetryCollector, resolve_job_context

async def sample(job_id: str):
    ctx = resolve_job_context(job_id)
    collector = TelemetryCollector(ctx)
    await collector.start()
    try:
        snapshot = await collector.next_snapshot()
        print(snapshot.to_json())
    finally:
        await collector.stop()

asyncio.run(sample("12345"))
```

## Output Formats

### JSON Lines (default)
```json
{"timestamp": 1705312234.567, "job_id": "12345", "hostname": "cn001", ...}
```

### CSV
```
timestamp,job_id,hostname,elapsed_seconds,cpu_cores,cpu_percent,cpu_effective_cores,...
1705312234.567,12345,cn001,3600,16,45.50,7.28,...
```

CSV rows are padded to a fixed 8-GPU column layout so every row has the same
number of columns.

## Limitations

- **NVIDIA-only** — AMD GPUs not currently supported
- **Single-node** — monitors only the local node; multi-node jobs show per-node data
- **GPU process isolation** requires running on the same node as the job (cgroup access)

## License

MIT
