# slurmwatch

Live, process-isolated node-local hardware telemetry for active Slurm jobs.

<p align="center">
  <img src="assets/demo.gif" width="840" alt="slurmwatch live TUI dashboard monitoring a Slurm job with CPU, memory, and GPU telemetry">
</p>

## Installation

```bash
pip install slurmwatch
```

## Usage

```bash
# Interactive TUI dashboard
slurmwatch <job_id>

# Headless logging mode (for batch scripts)
slurmwatch <job_id> --log metrics.jsonl &

# Auto-discovery mode (no job ID)
slurmwatch
```

## License

MIT
