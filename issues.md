# slurmwatch — Working Notes: facts-first UI + feature roadmap

_Rewritten 2026-07-07 against the **current working tree** on branch
`tui-trim-trends-bar-sw-alias` (HEAD `9726199`, plus an uncommitted "trim UI" batch in
`tui.py`/`test_tui.py`; 259 test functions). Every `file:line` below was checked by hand against
the tree as it sits on disk._

## Where we are

The big 2026-07-06 audit (the old §A–§E "GPU compute visibility / confusing & ugly / robustness"
plan) is **mostly shipped** — the 10 UI-clarity PRs + the v0.2.0→v0.2.3 releases landed labeled
bars, the distinct CPU/MEM/GPU palette, the two-bar GPU block (`compute` + `vram`), the JobInfoBar,
the honest-flat TRENDS bar, and the `sw` alias. The current branch is trimming the last clutter
(dropped the green "ALL HEALTHY" banner and the "CPU UNDERUSED" alarm, moved the TRENDS title into
the panel border, turned the sections into titled cards).

This document refocuses on two things:

1. **Part 1 — the concrete UI changes asked for this session** (facts-not-verdicts, the bar bug,
   the GPU one-line merge, the floating keybar, the drab palette). These are near-term, low-risk,
   and each is a rendering change over data we already have.
2. **Part 2 — the feature roadmap**: everything we could add to make the tool genuinely more useful,
   organized by theme, each grounded in where its data comes from (already-collected vs. one NVML/
   scontrol/cgroup call away) with a rough effort/value tag.
3. **Part 3 — carried-over robustness / test gaps** still open from the prior audit.

**Guiding principle the user set (applies to everything below): report facts, let the user judge.**
No "underused / overused / good / bad" verdicts anywhere. Allocated-vs-used, clocks, bandwidth,
stall %, error counts — state the number and the trend; the researcher decides what it means.

---

## Part 1 · Immediate UI changes (this session)

### 1 · Report facts, not verdicts — drop the judgement words

**Ask:** remove "underused" (and every word like it — "idle / throttling / high / near limit") from
the rows. The user can judge whether 12% CPU is fine; the tool should not editorialize.

**Where the words come from (all verified):**
- The three health functions each return `(level, word)`:
  - `_cpu_health` (`tui.py:286`) → `n/a` / `underused` / `healthy`
  - `_mem_health` (`tui.py:297`) → `near limit` / `high` / `healthy`
  - `_gpu_health` (`tui.py:305`) → `idle` / `throttling` / `active`
- The `word` reaches a **row** via `_status_suffix` (`tui.py:430-436`), called on the CPU row
  (`tui.py:457`), the MEM row (`tui.py:473`/`:479`), and the GPU header line (`tui.py:519`).
  `healthy`/`active` are *already* invisible (suffix returns `""` for level `ok`); the words that
  actually print today are **underused, high, near limit, idle, throttling** (plus an un-suppressed
  `n/a` edge case for a 0-core allocation, since level `none` ≠ `ok`).
- The `word` also reaches the **GPU device table** STATUS column: `status_txt` at `tui.py:576-579`,
  columns declared at `tui.py:547`/`:549`, written at `tui.py:588`/`:590`.

**Change (low effort):**
- Make `_status_suffix` (`tui.py:430`) **always return `""`**. The colored health **dot** already
  carries the level (green/amber/red ● ▲ ✖) — that's the honest signal; the *word* is the verdict.
  The `word` element of the health tuples then becomes vestigial at the call sites (`:452/:461/:504`)
  — keep the functions returning `level` for the dot color, drop the second element (or ignore it).
- GPU table: set `status_txt` to the **glyph alone** (drop the word at `tui.py:576-579`), or remove
  the STATUS column entirely (`:547/:549`, `:588/:590`) since the dot conveys it.
- Fix the `n/a`/`none` edge: level `none` currently isn't suppressed; either suppress it too or leave
  the dim `·` dot with no word.

**Banner is a separate decision (`_banner_segments`, `tui.py:318`).** The alarm strip mixes hard
facts with verdicts: `MEMORY 96%` and `2 OF 4 GPUS IDLE` are arguably factual (numbers/counts), but
`— OOM RISK` (`:330`), `— APPROACHING LIMIT` (`:332`) are judgements. **Recommendation:** keep the
banner as the "needs-action" strip but strip the verdict tail — e.g. `MEMORY 96% (limit 63 GiB)`
instead of `MEMORY 96% — OOM RISK`. Decide per-line; this is not a mechanical removal.

**Tests to update:** `test_tui.py:182-212` (health-tuple exact-word asserts), `:693-705`
(`underused` in the CPU row), `:722-739` (`throttling` in the GPU row), and any banner-phrasing
tests.

### 2 · Fix the RESOURCES-vs-TRENDS bar disagreement (the "4% bug")

**Symptom (reported):** the *same* MEM 4% draws an **empty** bar in RESOURCES
(`● MEM used ░░░░░░ 4% …`) but a **filled** bar in TRENDS (`MEM used 4% … ███░░░`). Same number,
two different bars — confusing.

**Root cause (verified): two un-shared bar helpers with different rounding + min-cell rules.**
- RESOURCES → `_labeled_bar` → **`_color_bar`** (`tui.py:158`). Fill = `int(percent/100*length)`
  (`tui.py:168`) — a **floor with no minimum cell**. MEM 4% on the 18-cell bar = `int(0.72) = 0` →
  **empty**.
- TRENDS → **`_trend_bar`** (`tui.py:683`). Its nested `cells()` (`tui.py:699-705`) uses
  `round(...)` **plus** a `max(1, n) if round(pct) >= 1` min-cell rule, on a ~74-cell bar. MEM 4% =
  `round(2.96) = 3` → **███**.
- Commit `9726199` only patched the `_trend_bar` path (min-cell gate), never `_color_bar` — so the
  two diverged for the 1–4% range. Even at equal width they'd disagree: at 18 cells, `int(0.72)=0`
  vs `round(0.72)=1→max(1,1)=1`. **The rule itself must be unified, not just the width.**

**Change (low effort):** extract one shared helper near `tui.py:149-176`:
```python
def _bar_cells(pct: float, width: int) -> int:
    pct = min(max(pct, 0.0), 100.0)
    n = min(width, round(pct / 100.0 * width))
    # A value that DISPLAYS as >=1% keeps at least one filled cell; a sub-0.5%
    # value that shows as "0%" draws empty so the bar matches its label.
    return max(1, n) if round(pct) >= 1 else n
```
- `_color_bar` (`tui.py:168`): `filled = _bar_cells(percent, length)`.
- `_trend_bar`'s `cells()` (`tui.py:699-705`): delegate to `_bar_cells`, **keeping** the
  peak-extension code at `:707-717` (that history tint is unique to TRENDS).
- Bonus consistency: `_color_bar`'s other callers gain the same rule — the GPU table bar
  (`tui.py:561`, width 8), the drill-in detail bars, and the JobInfoBar time bar. Desirable; the
  min-cell only fires when `round(pct) >= 1`, so a true 0% still draws empty everywhere.

**Add a regression test** asserting `_color_bar` and `_trend_bar` return the *same cell count at
equal width*, and that MEM 4% is non-empty in the RESOURCES bar (near `TestTrendBar`,
`test_tui.py:121`).

### 3 · Put the GPU `vram` line on the `compute` line, and tighten the layout

**Ask:** when there's room, render the GPU `vram` line on the *same* line as `compute` (they're one
device); rearrange for density; fall back gracefully when narrow.

**Current (verified):** `_gpu_block` (`tui.py:496-522`) emits **3 lines** per GPU:
1. `● GPUn` + status word (`:518-519`) — with §1 done, this line is just `● GPUn`, nearly empty.
2. `      compute ██████ 99%` (`:520`)
3. `      vram ████ 47%   67 / 140 GiB   555 W · 44 °C` (`:521`)

**Change (low effort):** `_gpu_block` already gets `bar_w`; also thread in the existing **`wide`**
flag (computed at `tui.py:444`, currently not passed). Then:
- **Wide terminal → one line per device:**
  `● GPU0   compute ████ 99%   vram ██ 47%  67/140 GiB  555 W · 44 °C`
  (fold the `GPUn` label, both bars, and the amt/pwr/temp onto a single row; shorten the bars a
  little so it fits 80–100 cols).
- **Narrow terminal → 2 lines:** header `● GPU0` + one combined `compute …  vram …` line, or fall
  back to today's 3-line block if even that overflows.

This removes ~1 line per GPU and reads denser, matching the "rearrange the layout" ask. **Note**
`test_tui.py:378` (`test_gpu_row_shows_compute_and_vram_separately`) asserts compute-line-index <
vram-line-index and that `W` is on the vram line — it **will break** and must be rewritten to check
"same line when wide, stacked when narrow".

### 4 · Make the layout fill honestly (the floating keybar)

**Question (reported):** the bottom key row (`q Quit  c CPU  m Memory  g GPU`) floats well above the
terminal floor with empty space beneath it — is the UI dynamic (will it get pushed down when there's
more to show), or is that gap a bug?

**Answer (verified): it's by design, and it *is* partly dynamic — but the slack pools below the bar,
so with little content the bar floats.** The `DashboardScreen` stack (`compose`, `tui.py:1110-1135`)
is Header (docked top) → `#banner` → `VerticalScroll #body` (resources + trends cards) → `#jobinfo`
→ `#keybar`. In the CSS (`tui.py:1033-1077`) **every** region is `height: auto` except `#keybar`
(`height: 1`, `:1073`); there is **no `1fr` and nothing docked** anywhere on this screen. So the
screen lays children out top-aligned, the stack's height = sum of content, and any leftover terminal
height collects as one empty band **below `#keybar`** (the last, un-docked flow child). As content
grows (more GPUs, banner alarms) the whole stack **does** push the keybar down, and once content
exceeds the terminal the *screen* scrolls (`DashboardScreen { overflow-y: auto }`, `:1034`). So:
more info → yes, it gets pushed down; little info → it floats. Downside of the current design: on a
short terminal the keybar sits **below the fold**, only reachable by scrolling.

**Recommended fix (one line, recommended):** change `#body` (`tui.py:1043`) from `height: auto` to
**`height: 1fr`**. The scroll pane then fills leftover height, dropping `#jobinfo`+`#keybar` to the
terminal bottom and keeping them **visible on short terminals** (scrolling moves to `#body`). Cost:
the empty band moves *inside* `#body` (below the trends card), and it breaks
`test_body_hugs_content_no_dead_space` (`test_tui.py:852-870`, which encodes the current gap==0
behavior) — update that test. To erase the visible band entirely, also give `#trends-panel`
(`:1049-1057`) `height: 1fr` so the TRENDS card stretches to absorb the slack. (Alternative: wrap
`#jobinfo`+`#keybar` in a `Vertical(id="bottombar")` with `dock: bottom` — pins the bar to the floor
while the body scrolls under it; wrap them together so their order doesn't invert.)

### 5 · Refresh the palette — it reads drab (measured, not taste)

**Ask:** the current colors are drab; make the UI more alive. All figures below were computed with
the **dataviz skill's `scripts/validate_palette.js`** (Machado-2009 CVD ΔE + OKLCH band + WCAG
contrast) against the card surface the marks sit on — re-run it after any tweak, and **mount +
eyeball** the TUI (the validator judges color, not layout).

**Why it's drab (measured):**
- **Washed block hues.** CPU `#4fb8cc`, MEM `#e08aa8`, GPU `#a884e0` (`tui.py:74-76`) all sit at
  OKLCH **L > 0.67** (above the dark band) → pastel/chalky on the dark surface. Worse, **CPU cyan is
  at the chroma floor (C≈0.10) — a near-gray**; the CPU bar/trend/label are the drabbest marks in
  the UI.
- **The 8-hue GPU cycle hard-fails CVD.** `_GPU_CYCLE` (`tui.py:91-100`) collapses `#5fe650`↔
  `#97cc47` to **ΔE 4.3 (deuteranopia)** — the green cluster the memory already flags. (8
  CVD-distinct hues on a dark surface is mathematically over-constrained; even the dataviz reference
  default fails all-pairs on dark.)
- **No elevation.** background `#1c1b1a` / panel `#1f1e1d` / surface `#262624` (`tui.py:124-126`)
  span only ~10 sRGB units, and cards use `$surface-lighten-1` + a coral border at **40% alpha**
  (`:1052`) — so page, screen, and card are one flat tone.
- **Grey everywhere.** `_DIM #a39b8d` / `_FAINT #6f685d` (`:63-64`) are low-chroma warm greys, and
  `_FAINT` is also the **empty half of every bar** (`:169`) — a large share of the frame is grey.
- **Stale theme comment.** `tui.py:52-61` claims "worst adjacent deuter ΔE 33"; that holds under
  deuter but is **11.1 under protan**, and coral chrome↔ok-green is **ΔE 1.9 under deuter**.

**Proposed refresh (drop-in hex — see the Appendix for the full block):** deeper, saturated block
trio `#159fc0 / #df5f97 / #8a6ee6` (+ vram `#cbb8f5`) that **passes all four validator checks**
(worst CVD ΔE 12.9, up from 11.1, and cyan leaves the chroma floor); a composite **4-hue × 2-shade**
GPU device cycle that lifts device CVD from 4.3 to ~12 (identity by hue *and* lightness, with the
numeric index cell as the guaranteed secondary channel); real 3-plane elevation (bg `#141312` /
surface `#1e1c1b` / lifted card `#262320` with a stronger `round $accent 55%` border); warmer,
readable dim text and a lifted empty-bar track so bars sit in a visible groove; and one considered
`_ACCENT_HI #ef9a72` for peaks. Keep the health hexes but keep relying on the **glyphs** (green↔red
is an unavoidable ΔE 5.9 under deuter). After changing, regenerate the ΔE numbers in the theme
comment so it stays honest.

---

## Part 2 · Feature roadmap

Organized by theme. Each item notes its **data source** — the biggest wins are cheap because the
data is *already collected* and merely hidden. Effort S/M/L, value high/med/low.

### A · Answer "is MY job using the hardware I was allocated?"

The single most load-bearing question, and the data mostly already exists.

- **[high · S] Per-job GPU compute share (JOB%) on the dashboard.** Render the job's own summed SM
  util + VRAM beside the device bar, e.g. `compute 82% (job 79%)`. On a shared/non-isolated node the
  device bar can be a neighbor's load; the gap between device% and job% is the whole question.
  _Data: already collected — `process_utilization_percent`/`process_memory_bytes`
  (`collector.py:730`/`:746`, `model.py:45-46`); shown **only** in the `g` drill-in table
  (`tui.py:584-588`), omitted from the dashboard `_gpu_block` (`:496-522`) and the overview table
  columns (`:549`)._
- **[high · S] GPU memory-bandwidth utilization.** A `mem-BW` bar next to `compute`: low compute% +
  high mem-BW% = memory-bound; low both = starved/idle. The cheapest possible second axis.
  _Data: `nvmlDeviceGetUtilizationRates(handle)` is **already called** at `collector.py:680` and only
  `.gpu` is read — the sibling `.memory` is discarded. Add a `memory_bw_percent` field
  (`model.py:34`) + CSV plumbing._
- **[high · S] Report the throttle REASON, not a yes/no.** `throttling: power cap` / `thermal` /
  `HW slowdown` instead of a bare boolean — the reason is what's actionable (raise power limit vs.
  cooling vs. failing card). _Data: `_check_gpu_throttling` (`collector.py:803-847`) already ANDs
  `nvmlDeviceGetCurrentClocksThrottleReasons` into a bitmask, then throws it away for a bool
  (`model.py:44`) — keep the matched reasons._
- **[med · S] "util unavailable (MIG)" honesty.** When `utilization_available` is False, label
  `compute: n/a (MIG slice)` and use VRAM occupancy as the proxy, instead of a bare bar that reads
  as real 0%. _Data: `utilization_available` already collected (`model.py:50`, `collector.py:764`),
  currently unread by the TUI._

### B · New views — the NODE / JOB tabs the user asked for

These need an enabling shell first, then each view is cheap.

- **[high · M] Tabbed shell (Textual `TabbedContent`).** Persistent, always-visible tabs
  `OVERVIEW · NODE · JOB · GPUs · PROC · TRENDS · LOG`, replacing the modal `c/m/g` drill-ins. A
  NODE/JOB view has nowhere to live without this. Auto-detected, no flags (remember the last tab).
  _Data: refactor `compose()` (`tui.py:1110`) — no `TabbedContent` import today (`tui.py:16`); reuse
  existing widgets as panes._
- **[high · M] NODE tab — host capacity & live pressure.** `CPUTot/CPUAlloc/CPULoad`,
  `RealMemory/FreeMem`, full `Gres/GresUsed`, node `State`/drain `Reason`/`ActiveFeatures`, uptime;
  plus the job's slice of each (job cores of node cores, job GPUs of node GPUs). The TUI already runs
  **on** the compute node after the srun hop, so the query is local. _Data: `scontrol show node
  <hostname>` parsed via existing `_parse_scontrol_field` (`slurm.py:495`); hostname already known
  (`collector.py:53-56`). New periodic collector path, off the event loop like `resolve_current_jobs`._
- **[high · M] NODE tab — co-tenant jobs on this node.** The other running jobs sharing the node
  (id/user/account/CPUs/mem/GRES), current job highlighted — names the contention behind your
  throttling / cache thrash / a "device busy" GPU that isn't yours. _Data: `squeue -w <hostname> -h
  -o '%i|%u|%a|%C|%m|%b'`; parser mirrors `resolve_current_jobs` (`slurm.py:155-199`)._
- **[high · M] JOB tab — full Slurm identity.** Account, QOS, Reservation, Submit→Start→Eligible
  times (derived queue-wait), Command, WorkDir, StdOut/StdErr, Priority/Nice, Requeue, Dependency,
  JobState, ExitCode. None of this is shown today (JobInfoBar has only id/user/partition/node/time).
  _Data: **already fetched** — `resolve_job_context` runs the full `scontrol show job -d`
  (`slurm.py:212`) and parses only ~15 of its fields; the rest is sitting in that string._
- **[high · S] JOB tab — TRES requested-vs-granted.** Render the job's TRES (cpu/mem/gpu/node/
  billing) as requested-vs-allocated — reveals silent under-grants, judgement-free. _Data:
  `JobContext.tres` is populated (`slurm.py:318`) but **read nowhere** in the TUI (confirmed dead
  data — also flagged §E). Add `ReqTRES` from the same scontrol output._
- **[med · M] PROC tab — the job's process tree.** Per-PID command, state, CPU%, RSS, threads, and
  per-PID GPU memory — turns aggregate cgroup numbers into "which rank is the straggler / who's
  eating RAM". _Data: `_get_job_pids` (`collector.py:773`) already enumerates the PID set every
  cycle; per-PID GPU mem via `nvmlDeviceGetComputeRunningProcesses` (`collector.py:718-730`). Add
  `/proc/<pid>/{stat,cmdline,status}` reads (`_read_pid_cpu_ticks` at `collector.py:917` exists)._

### C · Pipeline / bottleneck diagnostics (facts, no verdict)

"Why is the GPU hungry but idle?" — the input-pipeline signals, all cgroup-v2 native or one NVML call.

- **[high · M] PSI stall strip — cpu / memory / io pressure.** `some avg10` from the job cgroup's
  `{cpu,memory,io}.pressure`: the fraction of the last 10s the job was stalled waiting on each. The
  most direct "what am I blocked on" signal in the kernel — io-pressure high = I/O-bound; cpu high =
  dataloader-CPU-bound; mem high = thrashing. _Data: new reads under `ctx.cgroup_v2_path`
  (`model.py:184`, already walked for `cpu.stat` at `collector.py:498`); parse `some avg10=`._
- **[high · M] PCIe throughput (H2D/D2H).** Per-GPU RX/TX MB/s — high RX + low compute% is the
  classic "CPU can't feed the GPU" stall; near-zero RX + idle GPU = nothing being fed. _Data: new
  `nvmlDeviceGetPcieThroughput(handle, RX_BYTES|TX_BYTES)` in the sweep (`collector.py:617/:658`)._
- **[med · S] cgroup CFS-quota CPU throttling.** `nr_throttled` periods / `throttled_usec` share —
  "the scheduler is capping you below the wall you think you have", a real, invisible cause of GPU
  starvation. _Data: nearly free — `collector.py:498` already reads `cpu.stat` for `usage_usec`; the
  same file has `nr_periods`/`nr_throttled`/`throttled_usec`._
- **[med · M] Per-job disk I/O throughput.** Summed `rbytes/wbytes` deltas from the cgroup `io.stat`
  as read/write MB/s — pairs with io-pressure to confirm dataset/checkpoint I/O is the cause. _Data:
  new `io.stat` under `ctx.cgroup_v2_path`, differenced per poll (cgroup-v2 native)._
- **[med · L] CPU per-core / dataloader-worker heatmap (in the `c` detail).** Which allocated cores
  are actually pinned busy — 4 saturated + 12 idle = too few DataLoader workers; evenly-warm = a
  healthy pipeline. `effective_cores` is an aggregate that hides this shape. _Data: new per-CPU
  deltas from `/proc/stat` masked to `cpuset.cpus.effective` (under `ctx.cgroup_v2_path`)._
- **[low · L] InfiniBand / NIC throughput for multi-node jobs.** High interconnect traffic + low
  compute% = communication-bound scaling — often the real bottleneck for distributed training.
  _Data: node-wide `/sys/class/infiniband/*/ports/*/counters/{port_xmit_data,port_rcv_data}`, gated
  on `snap.node_count > 1` (`model.py:66`); node-wide, not cgroup-isolated (label it so)._

### D · The rest of the GPU device picture

Round out a GPUs tab; each is one guarded NVML call.

- **[med · M] SM clock vs max clock** — quantifies throttle severity (a card at 40% of max under a
  power cap loses far more than one at 95%). _`nvmlDeviceGetClockInfo(NVML_CLOCK_SM)` +
  `...GetMaxClockInfo` (cache max per handle at `_attach_handle`, `collector.py:249`)._
- **[med · S] Power draw vs enforced power limit** — `310 / 400 W (78%)`; draw pinned at the cap
  explains a power-cap throttle. _`power_watts` already read (`collector.py:697`); add
  `nvmlDeviceGetEnforcedPowerLimit`._
- **[low · S] Encoder/decoder (NVENC/NVDEC) util** — for video/vision decode pipelines where SM util
  reads low but the decode engines are the bottleneck. _`nvmlDeviceGetEncoder/DecoderUtilization`._
- **[low · M] ECC / Xid error counters** — a nonzero uncorrectable count is "the hardware, not me";
  lets the user file a node-drain ticket instead of debugging their model. _`nvmlDeviceGetTotalEcc
  Errors` / `...GetMemoryErrorCounter`, guarded (ECC may be disabled)._
- **[low · M] Cumulative throttled time (violation counters)** — catches intermittent thermal dips
  (3s every 30s) that never trip the instantaneous bool but still cost ~10% throughput.
  _`nvmlDeviceGetViolationStatus(POWER|THERMAL)`, differenced per poll._
- **[low · S] Fan speed / HBM memory temperature** — a hot HBM stack or pegged fan precedes a
  thermal throttle the core temp hides. _`nvmlDeviceGetFanSpeed`; mem temp via `nvmlDeviceGetField
  Values([NVML_FI_DEV_MEMORY_TEMP])`._
- **[low · S] GPU device model name** — `NVIDIA A100-SXM4-80GB` frames every other number (is 82 GB
  used of an 80 GB card the 80 GB SKU?). _Already collected: `GpuMetrics.name` (`model.py:37`,
  `collector.py:267`), serialized but never rendered._
- **[med · M] Cross-GPU straggler view** — flag imbalance across the job's GPUs (`GPU3 compute 4% vs
  others 90%`), the signature of a hung rank / unbalanced shard; the banner only fires when **all**
  are idle. _Pure computation over `snap.gpus` (`model.py:65`)._

### E · Memory / lifetime facts (no verdict)

- **[high · M] Allocation-vs-usage ledger.** A bare-facts table: cores allocated N → effective M
  (avg K over the job), memory allocated G → used U (peak P), GPUs allocated Q → R running job
  processes, plus `GPU-hours busy / allocated`. Exactly the "facts not 'underused'" the user wants.
  _Data: all present — `cpu.effective_cores/cores_allocated` (`model.py:9-12`), `memory.working_set/
  limit/peak` (`model.py:19-27`), `elapsed_seconds` (`model.py:62`); accumulate running means in
  `_update_widgets` (`tui.py:1175`)._
- **[med · S] Swap in use + kernel OOM-kill count.** Two facts near the MEM row: `memory.swap.current`
  (thrashing explains sudden slowdowns) and `memory.events` `oom_kill` (a nonzero count is the
  smoking gun behind a silently restarted rank — the OOM guard only *predicts* proximity today, it
  never reports a kill already happened). _Data: new reads under `ctx.cgroup_v2_path`._
- **[med · S] Cumulative CPU-time & since-start min/avg/max.** The 60s TRENDS window can't answer
  "how did this job behave overall"; a lifetime mean is the honest sizing number. _Data:
  `CpuMetrics.usage_ns` (`model.py:10`) is collected but shown only in the remote text summary
  (`cli.py:393`), never live; the drill-in already computes min/avg/max (`tui.py:991`) — widen the
  window._
- **[low · S] Live process / thread count.** `9 procs · 148 threads` — a distributed job showing 3
  of 8 ranks has lost workers; a loader stuck at 1 thread explains a CPU-starved GPU. _Data: free
  from `_get_job_pids` (`collector.py:773`) + `/proc/<pid>/status` Threads._

### F · Product / workflow features

- **[med · L] HISTORY / REPLAY of a prior `--log` run.** Open an existing CSV/JSONL log and scrub
  the whole run (play/pause, jump to peak) with the live widgets — for post-mortems ("when did it
  OOM?"). slurmwatch can already record headless (`--log`, `cli.py:565`) but there's no way to read
  it back. _Data: reader that reconstructs `TelemetrySnapshot` from `to_csv_row`/`to_json`
  (`model.py:83`/`:71`)._
- **[med · M] ALERTS log + notification.** A timestamped, scrollable log of state transitions
  (OOM-approaching→critical, GPU idle ≥ N min, throttling onset, node drained) with an optional
  terminal bell / OSC-9 desktop ping. Alerts live only in the transient banner today — you can't
  tell it OOM-spiked while you were away. _Data: `oom_guard_warning/critical` (`model.py:24-25`),
  `_gpu_is_active` (`collector.py:875`), `throttling` (`model.py:44`) + transition tracking._
- **[med · S] HELP / LEGEND overlay (`?`).** A modal explaining the color system, health glyphs,
  every column, and the keymap — the dense rows have no legend (old §B3). _Static content from
  existing constants._
- **[med · S] CONFIG / ABOUT panel.** Show the effective thresholds actually driving any flag (OOM
  warn/crit %, GPU-idle %, poll interval, history window) and the data source in use (cgroup v1/v2
  path, or remote sstat) — so a user can see *why* something is flagged, read-only, no flags. _Data:
  `SlurmwatchConfig` (`config.py:31-44`) on the screen (`tui.py:1088`); cgroup paths/remote flag
  (`model.py:184-201`)._
- **[med · S] SNAPSHOT / EXPORT.** A key that writes or OSC-52-copies a paste-ready report of the
  current state (job identity + all numbers) for a ticket/Slack. _Reuses `TelemetrySnapshot.to_json`
  (`model.py:71`) + `latest_snapshot` (`tui.py:1089`)._

**Implementation caveats (from the agents):** NVML per-poll cost — PCIe/clocks/violation/ECC each add
calls inside `_nvml_lock`; guard each individually (like the existing sub-queries at
`collector.py:679-708`) and consider a slower cadence for the expensive ones. Shared non-isolated
GPUs — device%, PCIe, power, IB are device/node-wide, so label them as such and always pair with
JOB%. MIG returns `NOT_SUPPORTED` for most rate/clock APIs — every new call needs the same
`contextlib.suppress(NVMLError)` guard. The `scontrol show node` / `squeue -w` subprocess features
should run on the same off-event-loop executor as `resolve_current_jobs`/`resolve_job_context` and be
throttled like the remote sstat cache (`collector.py:64-65`) so they don't hammer slurmctld.

---

## Part 3 · Carried-over robustness & test gaps (still open)

Low-severity, verified in the prior audit; none crash the happy path, but each is a real latent bug.

- **C1 · `stop()`'s graceful teardown-wait is dead code.** `fut = self._inflight_collect` is
  snapshotted **after** `await self._task` (`collector.py:87` vs `:82`); the loop's `finally` nulls
  `_inflight_collect` (`:318`) before then, so the bounded "let the in-flight collection finish"
  wait never runs. Correctness rests entirely on the NVML lock. _Fix: capture `fut` before
  `self._task.cancel()`._
- **C2 · `--interval inf`/`nan` bypass the finiteness guard.** `_positive_float` (`cli.py:124-131`)
  rejects only `<= 0`; `inf`/`nan`/`1e999` pass, then `asyncio.sleep(inf/nan)` misbehaves. The env
  path already guards this. _Fix: add `math.isfinite` check._
- **C3 · Config values accepted without range validation.** `validate()` (`config.py:57-77`) range-
  checks only the two OOM thresholds; `cpu_underuse_threshold` (expect [0,1]) and
  `gpu_idle_threshold` (expect [0,100]) and `csv_dialect` are unchecked, so nonsense values produce
  nonsensical results / a raw `csv.Error` deep in the output path. _Fix: range-validate + validate
  the dialect against `csv.list_dialects()`._
- **C4 · Misleading messages for bad string env vars.** An invalid `SLURMWATCH_ASCII` is reported as
  "expected a finite number" (`config.py:127-130`); `SLURMWATCH_FORMAT=JSON` silently emits CSV
  (exact-lowercase compare, `cli.py:352`). _Fix: bool-specific message; casefold + validate FORMAT._
- **E1 · `_gpu_is_active`'s preferred branch is untested.** Every test feeds
  `process_utilization_percent = 0.0`, so the job-util branch (`collector.py:874`) is never
  exercised — it could be inverted/deleted with the suite green. Add a case with process util above
  threshold, device util below, process VRAM < 50% of used. (This is exactly the §A signal.)
- **E2 · Dead data fields.** `JobContext.tres` and `min_memory_node` are populated (`slurm.py:307-
  308`/`:318`) but never read/serialized. **§B's TRES-requested-vs-granted view consumes `tres`** —
  do that, or drop the field.

---

## Suggested sequencing

1. **Part 1** — the facts-pass (drop verdict words §1), the bar-bug fix (§2), the GPU one-line merge
   (§3), the `#body: 1fr` layout fix (§4). All low-risk rendering changes; add/adjust the named
   tests as you go. Do the palette refresh (§5) as its own render-and-eyeball PR.
2. **§A** — JOB% on the dashboard + mem-bandwidth% + throttle-reason. All three are cheap (the data
   is fetched already) and directly answer "is my job using the hardware".
3. **§B** — land the `TabbedContent` shell, then the NODE tab (the user's explicit ask) and the JOB
   tab (nearly free from the scontrol output already fetched; consumes the dead `tres` field, E2).
4. **§C/§E/§F** — PSI stall strip + PCIe (the bottleneck story), then history-replay / alerts /
   help / snapshot as the tool matures.
5. **Part 3** — fold the robustness fixes in alongside, and add a regression test for each Part-1/§A
   change (the current suite would not have caught the bar-bug or the verdict words).

---

## Appendix · Palette proposal (drop-in) + validator

All hex validator-checked (dataviz `scripts/validate_palette.js`) on the card surface; re-run in both
CVD modes and **mount + eyeball** before committing (the validator judges color, not layout).

```python
# Block trio — deeper & saturated (tui.py:74-76, 81). Passes all 4 checks; worst CVD ΔE 12.9.
_CPU_COLOR      = "#159fc0"   # was #4fb8cc — deep cyan, off the chroma floor
_MEM_COLOR      = "#df5f97"   # was #e08aa8 — deeper rose, unmistakably not coral
_GPU_COLOR      = "#8a6ee6"   # was #a884e0 — deeper violet
_GPU_VRAM_COLOR = "#cbb8f5"   # was #d3c0f5 — lilac re-paired to the new violet

# GPU device cycle — composite 4 hue x 2 shade (tui.py:91-100). Worst all-pairs ΔE ~12 (was 4.3).
# Identity by hue AND lightness; the numeric index cell is the guaranteed secondary channel.
_GPU_CYCLE = [
    "#a98ff0",  # violet bright
    "#7658d8",  # violet deep
    "#3fc9d6",  # teal   bright
    "#1c8a97",  # teal   deep   (nudge to #1a8f9c to clear chroma 0.10, ~1 ΔE cost)
    "#e6b24a",  # amber  bright
    "#b07d1e",  # amber  deep
    "#ef8fc0",  # pink   bright
    "#c04f86",  # pink   deep
]

# Real 3-plane elevation (theme, tui.py:124-126) — was all within ~10 sRGB units.
background = "#141312"   # page plane
surface    = "#1e1c1b"   # the screen
panel      = "#262320"   # lifted card / bottom bars
# Card CSS (tui.py:1051-1052): background #262320;  border: round $accent 55%;   (was $surface-lighten-1 / $primary 40%)

# Warmer, readable text + a live accent (tui.py:62-66).
_INK       = "#ede7dd"   # slightly brighter primary
_DIM       = "#b3a998"   # warmer secondary (7.5:1 on the card)
_FAINT     = "#857d70"   # empty-bar track lifts to 4.0:1  (or split: keep _FAINT for text, add _TRACK="#3a352f")
_ACCENT    = "#d97757"   # keep coral chrome — the brand anchor
_ACCENT_HI = "#ef9a72"   # NEW brighter coral for peaks/highlights (verify it clears amber health)
```

Health hexes stay (`tui.py:110`); keep the `● ▲ ✖` glyphs — green↔red is an unavoidable ΔE 5.9 under
deuteranopia, so never encode health by color alone. Fix the now-stale ΔE claims in the theme comment
at `tui.py:52-61` after changing the hexes.

_Verification: the row/bar/health/GPU-block/layout/theme anchors above were each read directly from
the working tree (`tui.py` as it sits on disk, uncommitted "trim" batch included). The feature data
sources were located by a fan-out grep of `collector.py`/`slurm.py`/`model.py`/`cli.py`; the palette
figures were computed with the dataviz validator. No source files were modified in producing this
document._
