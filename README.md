<div align="center">

# 👀 slurmwatch

**Is your running Slurm job actually using what you asked for? See it live.**

<a href="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml"><img src="https://github.com/PursuitOfDataScience/slurmwatch/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
<a href="https://pypi.org/project/slurmwatch/"><img src="https://img.shields.io/pypi/v/slurmwatch.svg?cache=bust" alt="PyPI"></a>
<a href="https://pypi.org/project/slurmwatch/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/badges/downloads.json" alt="PyPI downloads per month"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
<img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
<a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/lint-ruff-261230.svg" alt="Ruff"></a>

<img src="https://raw.githubusercontent.com/PursuitOfDataScience/slurmwatch/main/assets/demo.gif" width="860" alt="slurmwatch live view: CPU, memory and GPU bars for each of the job's processes, a job summary card, and a time-budget bar. The memory row turns amber, then red, as it nears the limit.">

</div>

## ✨ Install

```bash
pip install slurmwatch      # or: uv tool install slurmwatch / pipx install slurmwatch
```

Python 3.10+ on Linux. GPU panels switch on by themselves on NVIDIA nodes.

## 🧰 Use

```bash
slurmwatch 12345    # watch a specific job
slurmwatch          # or auto-discover your running job
sw 12345            # "sw" is a short alias
slurmwatch --help   # everything else
```

| Key | Does |
|---|---|
| `c` `m` `g` | drill into CPU, memory, GPU |
| `o` `e` | follow the job's stdout, stderr |
| node number, `←` `→` | switch node |
| `p` | expand a truncated path |
| `q` | back, then quit |

## 📌 Good to know

🧍 **Only your processes count.** A neighbour on a shared node never inflates your numbers.

🧠 **Memory is your real working set**, shown against your `--mem`, so you see an OOM coming.

🎮 **An idle GPU still holding memory is flagged.** `nvidia-smi` cannot tell you that about *your* job.

⏳ **A pending job gets answers, not an error**: why it waits, when it should start, and a
`scontrol update` that moves it somewhere it fits.

🔌 **Run it from the login node or the compute node.** Where it cannot attach, it falls back to
an `sstat` summary.

## License

MIT
