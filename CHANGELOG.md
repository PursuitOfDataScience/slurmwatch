# Changelog

Every released version of slurmwatch, newest first. Each entry is the release note
that was written at the time, kept here because the tag it lived in may no longer
exist: the tag list had grown to one per release — thirty of them, several a
handful of lines apart — and it was thinned to the snapshots that differ
substantially from the one before. Nothing was actually lost in the thinning: the
commits are all still on `main`, every version is still on PyPI, and the notes are
all still here. Only the labels went.

Surviving tags are marked **(tagged)**.

## v1.2.2 — 2026-08-26 **(tagged)**

`1783cb69a844`

startup and the pending view, measured and cut; no surface changed

Performance only. Every number below is a median measured on midway3 (Slurm
20.11.8) against a 4647-job queue, with the controller quiet; each was taken
against a pristine `git archive HEAD` tree with the arms interleaved, because this
controller swings `scontrol show job` from 140 ms to 2.4 s and a non-interleaved
A/B showed the optimised arm four times SLOWER purely by luck.

Startup, on every invocation
  * **first paint 575 -> 317 ms, first data frame 795 -> 550 ms.** Two causes. The
    version string was resolved at import through `importlib.metadata`, which pulls
    email, zipfile and importlib.resources behind it — ~34 ms of a ~130 ms cold
    start, on every run, for something only `--version` prints. And importing
    textual (~120 ms) ran strictly after the two controller round-trips that
    resolve the job, rather than inside them; a side thread now overlaps the two,
    which is free because those round-trips are subprocess waits.

The pending view — the slowest path in the tool, and the largest win
  * **first full frame ~3.7 s -> ~1.2 s, and one fewer Slurm call.** Its three
    resolves (partitions, queue counts, priority rank) depend on the job and on
    nothing from each other, but were awaited in sequence, so the view waited for
    their sum where the slowest term alone is ~2.1 s. And the QOS association
    lookup ran *on the event loop inside a render helper* — a ~222 ms blocking
    `sacctmgr` on every redraw, twice on the first paint. It is DB configuration,
    so it is read once per process now; a transient failure is deliberately not
    cached, because "unknown" permanently softens the view's advice.
  * **the text renderer had the identical defect: 1065 -> 748 ms.** Same fix, with
    a thread pool rather than `asyncio.gather` since that path is synchronous. The
    report's output is byte-identical and its line order is pinned by a test.

The off-node hop
  * The GPU-capability probe — a whole extra `srun` step creation before the real
    hop — ran even for jobs holding no GRES, where its answer cannot be used: the
    ssh escalation it feeds is gated on the job having GPUs, and `--gres=none` is a
    no-op without them. CPU-only jobs attach directly now. Measured at 168 ms for
    the probe itself; the end-to-end effect is not demonstrable, because that path
    swings 1.3-5.2 s on `srun --pty` step-launch variance alone.

What was deliberately left alone
  * The ~130 ms `scontrol show job` RPC. It is the single biggest block in startup
    and no flag variant is cheaper — `--local` and `-o` are noise, `--json` does not
    exist in 20.11 — and it cannot be dropped, being the only source of GRES `IDX:`,
    the stdout/stderr paths and the array-task fields.
  * The 200 ms window before the first sample. `_collect_cpu` reports 0.0% with no
    baseline to difference against, so publishing a frame early would publish
    CPU=0 — the exact reading `--once` calls out as the dangerous one. Painting
    memory and GPU early instead would need a real "CPU pending" state in the model,
    the CSV header and the JSON payload; that is a schema decision, not a speedup.

Guards
  * Every optimisation here was invisible to the existing suite — it passed with
    each one reverted. Seven mutations were tried and all seven are now caught, one
    test each. The concurrency guards use a `threading.Barrier` with a timeout, so a
    return to serial calls fails outright instead of merely running slow.
  * 1703 tests.

## v1.2.1 — 2026-08-25 **(tagged)**

`4c107726e1dc`

verified on two more Slurm clusters; six defects the new ones exposed

Ran the tool on Booth's mercury (RHEL 9, Slurm **25.11.3**, cgroup v2, 25GbE
RoCE) and pythia (RHEL 8, Slurm **24.11.5**) alongside midway3's **20.11.8** — a
five-year spread of Slurm. Bug fixes only; no field added or removed on any
surface.

Wrong answers the new clusters produced
  * **a cancelled array task read as RUNNING, on a sibling's node.** Slurm prints
    `JobId=<the array BASE>` for a task that has lost its allocation, so the facts
    query widened to the whole array and took the first row it got: the task
    reported the wrong state, the wrong node, and a sibling's CPU/node counts as
    its own denominators. squeue's `%i` is now asked for and matched, and the
    task's identity comes off the record's first line — the one stretch of output a
    newline in the job name cannot forge lines ahead of
  * **a node was abandoned because ssh was refused there.** The ssh transport was
    preferred over a monitor step on `which("ssh")` alone, which proves the client
    exists and not that the site permits the login. Where it does not, ssh answers
    "Permission denied" — the same words a refused Slurm step uses — so the node
    switcher gave up for the rest of the session on a node the step could still
    have streamed, and the banner blamed Slurm. The rung is recorded at launch, ssh
    is retired per node, and the step gets its turn
  * **a finished job's machine row dropped the state it finished in.** `--once
    --json` emitted every field null beside prose that named the state, so a poller
    watching a job through its lifecycle lost COMPLETED / CANCELLED / TIMEOUT at
    exactly the frame it mattered. The state (and name, owner, partition) now
    travel with the error
  * **a partition that can never run the job was labelled with a transient
    blocker.** A GPU-less partition read "no GPU" or "no room" depending on
    whether it happened to have an idle node that minute — the same hardware and
    the same request giving two verdicts, one of which invites a wait that cannot
    end. Permanent misfits are decided before scarcity

Noise and wrapping
  * the ssh hop no longer leaks ssh's own `Permission denied (publickey,…)` onto
    the terminal — it runs on a PTY, so there is no second stream to redirect;
    silenced, restored under `-v`, and the cause stated in slurmwatch's words
  * the foreign-job "No Live View" note keeps its indent when it wraps

Also landing here, accumulated since v1.2.0
  * control characters in a job name are neutralised on all three interpreters of
    that field: `sbatch -J $'\e[2J'` cleared the dashboard and `cat` executed it
    out of the CSV
  * `cpu_source` says which counter produced `cpu_usage_ns` ("v2"/"v1"/"proc"/
    "sstat"/"mock"); measured on a live reservation the /proc sum read 127,053
    CPU-s where `sacct TotalCPU` said 612 s, and nothing on the row explained why
  * the fabric rate is measured on a monotonic clock, with a short-window floor —
    wall-clock could divide a real byte delta by an NTP step and publish a
    throughput above the link's ceiling
  * `sw --help` survives a stream that cannot encode an em dash, and says once why
    the output looks plainer
  * "no jobs found" is not the answer while squeue is still showing a COMPLETING
    or SUSPENDED job
  * partition suggestions are filtered by association as well as capacity, and the
    remedy command carries the job id and QOS so it runs as printed
  * one `--log` file mode, so the umask no longer decides it
  * `--demo` node switching visibly changes the numbers

Tests: 1563 → 1694, four gates clean on all three clusters, and each of the six
fixes mutation-verified (20 deliberate reverts, 20 caught). Not exercised there
and unchanged apart from the stream fix: real GPU telemetry and multi-node
fabric/node-switching — mercury's H100s were held for hours by other users, every
mercury QOS caps a job at one node, and pythia grants this account no association.

## v1.2.0 — 2026-08-24 **(tagged)**

`cdf0b732d726`

cluster-agnostic hardening, verified on a second Slurm cluster

Worked through a cross-cluster portability review and then ran the result on a
genuinely different machine (Slurm 25.11, cgroup v2, RoCE, `python3` 3.9) rather
than only reasoning about it here. Additive JSON/CSV fields, no removals.

Honesty of the numbers
  * an unreadable GPU set no longer reports "0 active"; unread means null
  * an off-node reading Slurm has not sampled yet is no longer published as a
    zero: snapshots carry `usage_sampled`, and `--once` emits the no-telemetry
    shape instead
  * rows say how old their measurement is (`usage_age_seconds`) — off-node, most
    rows were re-serialisations of one sstat sample and nothing said so
  * the off-node OOM guard no longer claims it can only fire late: MaxRSS is a
    per-process sum, measured 13% above the cgroup's own cache-inclusive peak, so
    every surface that turns it into advice says it can overstate

Diagnoses that were not true
  * no Slurm on PATH said "couldn't reach the controller, it may be busy — try
    again"; it now says the tools are missing and retrying will not help
  * a stream step that cannot launch reported the same guess forever: its stderr
    is kept, summarised, and a permanent failure stops the retry loop
  * a failing telemetry read was reported as "Cannot write log file" and exited 1,
    ending a days-long `--log` run over one bad cycle on a perfectly writable file
    — and with a blank reason, since `str(TimeoutError())` is empty
  * a cancelled `--log` run could discard the very cancellation meant to stop it —
    unkillable rather than slow — and three teardown joins had no bound at all;
    both idioms now live in `aio.py`
  * pending reasons audited against two live queues: per-JOB limits are no longer
    called usage caps, account/user/group-scoped limits now are, and
    BadConstraints / InvalidAccount / InvalidQOS / JobArrayTaskLimit /
    MaxBillingPerAccount / "launch failed requeued held" get real explanations
  * the capacity table is suppressed whenever capacity is not the constraint, so
    the tip and the table can no longer contradict each other
  * the job picker's TIME column no longer drifts by the app's uptime, and a
    queued job's prose stays out of the machine-readable stream

Interfaces
  * job ids in the forms Slurm's own tools print: an array range `12345_[1-9%3]`
    and a step `12345.0` resolve to the right job instead of being refused
  * `--ascii` covers the framework's chrome too (panel borders, scrollbar, the
    drill-in figure): a pty capture now shows zero non-ASCII bytes
  * signals are one convention: `q` and a typed ctrl-c exit 0, a signal exits
    128+signum, and `--log` handles SIGHUP gracefully instead of dying mid-drain
  * the foreign-job view states what the job asked for, the only resource fact
    available across users
  * `--help` names every accepted id form and the hop timeout knob
  * RoCE ports no longer display an InfiniBand speed grade

Robustness
  * the dashboard and headless loops always yield: a loop that stops yielding is
    not slow but unkillable, since loop-registered signal handlers replace the
    default disposition and can never run
  * shared display rules live in `units.py` so one renderer cannot drift from the
    other

Tests: 615 → 1543, each fix mutation-tested rather than assumed covered, and the
whole matrix (3.10–3.13) green — 3.10 matters here, because `asyncio.TimeoutError`
is not the builtin `TimeoutError` before 3.11.

## v1.1.1 — 2026-08-04 **(tagged)**

`b292e657c8e4`

remediate an independent scan's 8 findings + 6 minors

Bug fixes only, no schema or CLI-surface changes:
  * an unreadable GPU VRAM read no longer renders as a measured 0% / 0 GiB
    (and no longer drags the drill-in chart's min/avg to zero)
  * unreadable readings stay inside their columns; mixed-capacity GPU nodes
    align their VRAM totals
  * the plain-text pending report no longer contradicts its own free-node and
    idle-core columns, and no longer truncates the partition table silently
  * the 'empty nodes' column header is honest for GPU jobs on clusters that
    expose per-node free GPUs
  * bare 'sw' (job picker) now shows another user's job as the read-only facts
    view instead of a doomed live collector, and a remote first node arms the
    switch banner/watchdog instead of hanging on 'awaiting telemetry'
  * --json no longer warns 'ignoring' on the very path that honours it
  * cgroup-v1 memory falls back to /proc RSS like v2 already did
  * cpu_usage_ns reports the last known counter instead of a fake reset
  * --append warns when the target CSV has fewer GPU columns than the job
  * a job name containing 'JobId=1+1' no longer looks heterogeneous

## v1.1.0 — 2026-07-29

`6fd011226d65`

multi-agent bug audit remediation

Fixes found by a 10-agent independent audit of the whole package. Highlights:

- NVLink throughput was measuring ONE LINK, not the fabric: the THROUGHPUT_DATA
  counters are per-link and selected by nvmlFieldValue_t.scopeId, which pynvml only
  sets for a (fieldId, scopeId) tuple. Reported rates were 1/18 of reality on an
  18-link H200, so a saturated NCCL ring looked idle.
- The cgroup-v2 OOM guard measured the job against its own --mem REQUEST instead of
  the kernel's kill point, producing a false 'near limit' critical that v1 correctly
  avoided for identical machine state.
- CPU rates could be differenced across three incomparable counters, yielding
  effective_cores=982 on a 4-core job and latching that into the peak.
- The pending view could advise a requeue onto a partition with zero free GPUs, and
  a fixed-width sinfo parse could read a fully-allocated GPU node as entirely free.
- 'sw <job> > file' blocked forever in an invisible TUI; it now emits one snapshot.
- Unreadable GPU VRAM/power/temperature were published as measured zeros, scoring a
  99%-busy GPU 'idle' and rendering a MIG slice at '0 W / 0C (32F)'.

New (additive): CSV gains cpu_usage_ns, gpu_monitoring_available,
gpu_<N>_throttle_reasons and gpu_<N>_mem/power/temp_available. Parse the CSV by
HEADER NAME, not column index — the fixed and per-GPU blocks both grew.

813 tests, all four gates green on py3.10-3.13.

## v1.0.1 — 2026-07-29

`2f2f82ffb893`

count free GPUs per node instead of only fully-idle nodes

## v1.0.0 — 2026-07-26 **(tagged)**

`1ea5e23099a0`

first stable release

## v0.10.1 — 2026-07-26

`a838a1c1d1ba`

GPU temperature also reads in °F beside the °C NVML reports

## v0.10.0 — 2026-07-25 **(tagged)**

`b3d77438012b`

real-NVLink correctness, CUDA ordinals, and the job name everywhere

GPU / NVLink (first validation on real NVLink hardware — a 3x H200 SXM node;
every prior release was checked only on PCIe H100s, so this path had only ever
run against mock NVML):
- the fabric generation is no longer read from nvmlDeviceGetNvLinkVersion, which
  returns a driver-internal code, not the marketing generation: a live H200 on
  driver 535 answers 7, and the header printed "NVLink 7". It now comes from the
  device model, so an H200 reads NVLink 4.
- per-link and per-GPU bandwidth no longer vanish when the driver reports
  SPEED_MBPS_COMMON as NOT_SUPPORTED; the drill-in shows 25 GB/s per link and
  900 GB/s per GPU, matching NVIDIA's published H200 figure.
- live NVLink transfer no longer reads a hard 0.0 on the first sample, which made
  --once report an idle fabric on a job that had moved 90 GB.

Devices are now labelled by CUDA ordinal — the number the job's own code
addresses (cuda:0) — instead of NVML's device index, which is only the same thing
on a cluster with device-cgroup isolation. The drill-in names the nvidia-smi index
too when the two differ. New gpu_<N>_cuda_ordinal CSV column.

The job name (sbatch -J) is now shown wherever the job id is: the header leads
with it, plus the bottom bar, the JOB card, the foreign-job view, --json and CSV.
It had been parsed for the job selector and the pending view only, so the screen
you actually watch could never say which experiment it was.

CSV is additive only: fixed columns 24 -> 25 (job_name), per-GPU 15 -> 16
(gpu_<N>_cuda_ordinal). --append to a log from an older build warns on stderr.

776 tests; ruff + mypy clean on py3.10-3.13.

## v0.9.9 — 2026-07-24

`243dd439f4d9`

GPU devices are now named, plus the third-pass audit fixes and test coverage
for seven fixes that had none.

- feat(tui): each GPU device row carries its model — "● CUDA 0 · H100 PCIe" —
  so the power/VRAM numbers beside it have a frame of reference. NVML's product
  name is trimmed to the model (vendor and pure memory-size tokens dropped,
  PCIE -> PCIe) while the form factor is kept, since PCIe vs SXM4 is a very
  different power/bandwidth class. Padded across mixed-device nodes, dropped on
  a narrow terminal, and escaped for Textual's markup parser.
- fix: the interconnect line printed "0.0 GB/s" while data was flowing — any
  aggregate under 50 MB/s rounded away. It now drops to MB/s below 0.1 GB/s,
  in both the dashboard head and the drill-in.
- fix: peak_effective_cores was always 0.0 off-node beside a non-zero
  effective_cores, so a --json consumer sizing --cpus-per-task read "used no
  CPU". The peak >= current invariant now holds on both paths.
- fix: CSV gained cpu_peak_effective_cores (fixed columns 23 -> 24, additive) —
  it carried both memory peaks but no CPU peak, so CSV consumers could not do
  the right-sizing --json consumers could.
- fix: --demo now shows the enforced power cap ("used / cap W") the README GIF
  advertises, instead of a bare wattage.
- test: mutation testing found seven already-shipped fixes with no test at all,
  including the shared-node CPU over-report. All are now guarded; a
  46-mutation sweep across every past finding fails at least one test for every
  reversion.

pytest 735; CI green on Python 3.10-3.13.

## v0.9.8 — 2026-07-24

`8006579cd35d`

Second-pass number-accuracy audit remediation + two UI fixes.

- fix: reap the node-stream srun subprocess on quit (no more "Event loop is
  closed" on exit)
- ui: per-label GPU-block colours (green active / amber idle, coral TX / teal
  RX arrows, ink fabric label)
- A1 NVLink live throughput uses a monotonic clock (no clock-step spike)
- A2 GPU compute shows "n/a" (not a false 0%) when NVML util is unreadable
- A3 effective_cores reveals CPU over-subscription (uncapped), with a min-dt
  guard so a bogus-window spike can't latch into the peak
- A4 off-node per-node peak uses ceil (OOM-safe for --mem)
- A5 _format_bytes promotes on the rounded value (no "1024.0 GiB")
- A6 queue/array counts include RESIZING/SIGNALING/REQUEUED
- A7 MIG vs transient util failure split via utilization_supported
- working_set_percent (cache-excluded) added to --json/CSV
- additive CSV columns: mem_working_set_percent, gpu_N_util_supported

## v0.9.7 — 2026-07-23

`1989ff1b8b2a`

number-accuracy fixes + GPU power cap

- CPU: no shared-node over-report (job-scoped /proc fallback vs node-wide cpuacct)
- memory: cache-excluded 'peak working set' for --mem sizing; OOM guard vs true kill-point
- pending: per-partition rank + job-id-deduped counts for -p a,b
- GPU: power shown vs enforced cap ('used / cap W'); throttle_reasons + CSV util_available

## v0.9.6 — 2026-07-23

`73d5216e87a8`

GPU interconnect view for multi-GPU jobs

For a multi-GPU-on-one-node job, show how the devices are wired and how
much data is moving over the fabric: NVLink generation + per-link/aggregate
bandwidth, the pairwise NVLink/PCIe topology grid (a la nvidia-smi topo -m),
and live data-transfer speed. Surfaced in the GPU drill-in (g), on the main
dashboard GPU head, and in --once --json as an "interconnect" object.

- GPU interconnect view: NVLink gen + speed, topology matrix, live traffic
- Live transfer also on the dashboard head (up = tx, down = rx, GB/s)
- Per-device throughput stored at ~1 MB/s precision so summed traffic on
  several devices is not rounded away to 0

## v0.9.5 — 2026-07-22

`ee2da150466e`

remove the dynamic status banner (redundant with the resource rows)

## v0.9.4 — 2026-07-20

`ed30b6f858b3`

Robustness + cluster-agnostic hardening since 0.9.3:
- tui: single heavy selector frame; teardown robustness + honest edge cases
- cli/slurm: name-independent cgroup discovery + ssh transport ladder (PATH/SLURM_CONF, csh-safe)
- gpu: honest message when NVML unavailable vs GPU held by srun
- pending: heterogeneous/reserved/flagged/hidden partitions
- collector: clamp off-node mem %, /proc RSS fallback when no memory cgroup
- slurm: P mem suffix, het +offset cgroup id, normalized subprocess env

## v0.9.3 — 2026-07-18

`f93d9c75e09b`

Robustness (the headline — fixes latent transient-controller-failure bugs
present since 0.9.x):
- Selector cursor no longer freezes on a busy login node (poll moved to an
  exclusive worker off the message pump).
- No more bogus "Job X is in state 'RUNNING', not PENDING" / black screen when
  scontrol transiently times out (_sacct_final_state is terminal-state-only;
  resolve_job_context/resolve_pending_job re-raise transient errors).
- Broad audit batch: crash (empty job list), hang (unbounded --pty hop),
  array liveness scan, per-node GPU/capacity for pending, monotonic elapsed,
  foreign-job gating, markup/NaN/duration hardening.

Features & polish:
- Read-only styled view for another user's job (honest fallback, no srun leak).
- Array-membership display.
- Job selector: launch colour flourish (plays once), crisp translucent double
  border, responsively sized panel; --ascii/demo/first-run/--help polish.

## v0.9.2 — 2026-07-16

`6d3183bf567f`

Match the validated 0.9.0 module: revert the v0.9.1 JobInfoBar id change
(back to v0.9.0 behaviour) and quit the job picker silently (drop the
'No job selected.' note).

## v0.9.0 — 2026-07-16 **(tagged)**

`549408cf30b6`

Job selector & GPU view overhaul.

Job selector (sw with no args):
- Refreshes its list live — newly submitted jobs appear and finished ones
  drop out without quitting; TIME column ticks in real time.
- Bigger, with column headers; the cursor tint keeps the RUNNING/PENDING
  tags legible; "TIME / WHY" header only when a job is actually pending.
- Quitting a job's view returns to the selector, landing the cursor back
  on the job you last opened.

Dashboard:
- GPU drill-in shows broad compute + VRAM charts for every device (was a
  duplicate per-device table); VRAM chart stats in GiB. "VRAM" cased
  consistently across the UI.
- GPU view: compute + VRAM bars per device, coloured by metric.
- Lifetime peak working set (MEM) and peak cores (CPU) for right-sizing
  --mem / --cpus-per-task.
- Header shows the job id you selected (array form) instead of the raw
  snapshot JobId, matching the JOB card.

## v0.8.2 — 2026-07-14

`8c5fa1a7333f`

JOB-card & GPU-view UI refinements

- feat(tui): single-GPU drill-in (g) draws the big filled history chart, like CPU/MEM (#84)
- feat(tui): JOB card shows the stdout/stderr log paths from scontrol (#85)
- feat(tui): JOB-card paths pack two-per-row on a wide terminal; p reflows to full-width (#86)
- fix(tui): drop the GPU-throttling banner alarm — it's a fact in the GPU view's STATUS (#87)

## v0.8.1 — 2026-07-14

`1d0237ccfb8b`

Drop the always-on login-node monitor-step note; keep only the
contextual "a launch looks stuck" escalation.

## v0.8.0 — 2026-07-14 **(tagged)**

`440a3b4987c2`

Login-node monitoring, pending-view overhaul, and hardening.

- Warn when the login-node hop's monitor step can block a new srun/mpirun.
- Pending view (why/when/where): accessible queue position, per-section colours,
  Title-Case headers, request details, account-filtered + capability-aware WHERE
  (names the specific blocker), animated "calculating"/"imminent" estimate, and
  pending jobs in the auto-discovery picker.
- Robustness: non-UTF-8-safe Slurm/proc decoding, monotonic CPU clock, bounded
  node-switch stream, exclusive/GPU/time-limit/per-node fit accuracy, thorough
  --ascii purity, and env-parse hardening — from three adversarial audit rounds.

## v0.7.0 — 2026-07-12

`207a75c6b184`

GPU views, reworked:
- Drill-in (g) charts every device inline (per-row TREND sparkline), no cursor.
- GPU is a first-class dashboard resource: a "● GPU  N devices · M active"
  section head aligned with the CPU/MEM rows.
- Colour is decorative resource identity, not a health verdict — the tool
  reports facts (cores, %/bars, ranges, temp, VRAM, status words) and lets you
  decide how the job is running.
- Roomier bottom info bar; a quiet, thin scrollbar that only appears on overflow.

## v0.6.0 — 2026-07-12

`b45592ea72e7`

hardening release

A batch of correctness fixes from two rounds of adversarial auditing. No new
features; the dashboard, pending view, and detail screens are unchanged in
layout — they're just more robust on real clusters and odd inputs.

TUI
- A job/GPU name containing '[' no longer crashes the dashboard (markup escape).
- GPU detail STATUS/JOB-VRAM cells no longer clip on a state-word change.
- Stream reconnect backoff can't overflow; a fixed race after a multi-node job
  ends no longer lets node-switch keys act on a frozen screen.

Collector
- /proc CPU fallback (cpuset-only nodes, no cpuacct cgroup) is now a monotonic
  accumulator, so fork-churn jobs (make -j, pipelines) no longer read 0%.
- That accumulator no longer double-counts a PID that briefly drops out of the
  sampled set (which produced a spurious CPU spike), and it bounds its memory.

Slurm parsing
- A crafted JobName can no longer shadow later scontrol fields; hostlist
  expansion is capped so a huge range can't blow up memory.

Pending view
- A job submitted to several partitions (-p a,b) now flags each of them as
  "your partition", and the no-capacity tip fires correctly.

Headless --log
- Ctrl-C / SIGTERM no longer hangs when the log sink stalls (full pipe, wedged
  NFS mount): the write is raced against shutdown and hard-exits if truly stuck.

Docs
- README documents `slurmwatch --demo pending`, the c/m/g drill-in detail
  screens, and the GPU-detail device picker.

## v0.5.0 — 2026-07-11 **(tagged)**

`bf8e13289b50`

New: pending-job insight — point slurmwatch at a queued job and get why it's
waiting (plain-English Slurm reason), when the scheduler estimates it'll start,
and where in the cluster it could run now (per-partition free capacity + a
"fits now?" check with the exact scontrol requeue command).

Also since v0.4.0: a large body of correctness fixes to per-node limits, the
remote sstat path (peak-vs-current, OOM), CSV logging (GPU columns, node
identity, append width), config validation, the node-switcher (SLURM_CONF,
post-job-end lockout, remote trend window), and a redesigned CPU/Memory/GPU
detail drill-in (big figure + tall history graph; readable GPU STATUS words;
a focusable, chart-driving device cursor).

## v0.4.0 — 2026-07-09

`2dc9acf4c189`

multi-node node switcher + dashboard hardening

Highlights
- Node switcher: type a node number to jump straight to any node (e.g. 199 of a
  200-node job), or step with ◂ ▸. Animated "switching to node N" banner that
  clears when the node's data lands, with a stuck-node warning.
- Robustness: srun stream detaches stdin (no more stolen keystrokes), backs off
  on a dead node, and tolerates Slurm NodeName != gethostname.
- Accessibility: short terminals keep the RESOURCES gauges on screen, width-aware
  footer, ASCII mode leaks no Unicode, per-card frame colours.
- Elided command/workdir paths with a "p" toggle to reveal the full path.

336 tests; CI green on py3.10-3.13.

## v0.3.0 — 2026-07-08

`a9632dbc9bbb`

facts-first dashboard

Since v0.2.3:
- Facts-first UI: dropped verdict words (rows/table/banner show the coloured
  health dot + the numbers, no "underused/idle/throttling"); one shared bar rule
  so the gauges and trends agree; folded the 60s range onto each row and removed
  the duplicate TRENDS panel; GPU compute+vram merge onto one line when wide; the
  job-info + key bar dock to the terminal floor; refreshed CVD-safe palette with
  real card elevation.
- New JOB card — account · QOS · state, command, workdir, submit→start queue
  wait (colourful, grouped), parsed from the scontrol record already fetched.
- Fixes: the empty-vs-filled bar bug, "ends by" (not "ends ~"), a GPU no longer
  reported idle AND throttling at once; robustness (collector teardown wait,
  --interval inf/nan, config range validation, case-insensitive SLURMWATCH_FORMAT).
- Docs: README + demo GIF refreshed for the new UI.

## v0.2.3 — 2026-07-06

`25b79fa1c89e`

steady trends show level + ripple; concise hop message

- Steady TRENDS lines draw at their absolute height again (so 13%, 4% and 99%
  sit at different heights) with a small travelling ripple so the band still
  visibly moves instead of sitting dead-flat. Moving series still auto-scale.
- The login-node srun hop shows a short "⠿ connecting to <node> …" line instead
  of the old verbose two-clause sentence.

## v0.2.2 — 2026-07-06

`08cc8bbc45d4`

quieter NVML warning + livelier trends

- No more scary "NVML Shared Library Not Found" WARNING on CPU-only / driverless
  nodes: a CPU-only job skips NVML entirely, and a genuine missing-driver case is
  a quiet INFO ("No NVIDIA driver on this node; GPU monitoring off").
- TRENDS lines now auto-scale to their own range so a small real fluctuation
  reads as a gentle live wave instead of a dead-flat row; a truly constant series
  is still labelled "steady".

## v0.2.1 — 2026-07-06

`d5fe5575b5a7`

steady trend band shows the level

Fix: a "steady" TRENDS line was drawn at a fixed mid-height, so a GPU pinned at
99% looked identical to memory steady at 3%. Steady bands now render on the
absolute 0–100 scale, so the band's height reflects the value (99% = nearly
full, 3% = thin low). Moving series still auto-scale to their own range.

## v0.2.0 — 2026-07-06 **(tagged)**

`fae6e48bbb85`

dashboard UX overhaul

A batch of live-dashboard clarity/readability improvements:
- GPU row shows compute (SM) vs VRAM as two clearly labeled bars
- distinct, CVD-safe per-block palette (CPU cyan / MEM rose / GPU violet)
- per-GPU device colours in the multi-GPU table (up to 8, with row gaps)
- "Recommendations" panel: actionable flags only, no context-free "good"
- labeled bottom job bar with a live time-budget (elapsed / limit / left /
  end), coloured by urgency
- TRENDS: labeled, auto-scaled sparklines so steady lines aren't flat/redundant
- per-key coloured footer; cleaner banner; removed stray header icon

261 tests, ~88% coverage; CI green on Python 3.10–3.13.

## v0.1.2 — 2026-07-03

`2de6fce90443`

_The tag carried no note beyond the version._

## v0.1.1 — 2026-07-03

`aeae54a27227`

_The tag carried no note beyond the version._

## v0.1.0 — 2026-07-03 **(tagged)**

`485b5149d97e`

_The tag carried no note beyond the version._
