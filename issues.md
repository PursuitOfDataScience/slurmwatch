# slurmwatch — Multi-Agent Bug Audit + Remediation

**Date:** 2026-07-29
**Version audited:** working tree == PyPI/module **v1.0.0** (`d1d4b8b`)
**Method:** 10 independent read-only auditor agents, each with a distinct scope and no
knowledge of the others' findings, followed by adjudication. A finding counted as real
only if corroborated — by a second agent, by my own independent re-derivation from the
source, by primary vendor documentation, or by an empirical reproduction. Findings that
were *demonstrated* (mutation applied → suite still green; command run → wrong output
captured) needed no vote: they are facts, not opinions.

**Baseline:** 786 tests, all four CI gates green.
**Now:** **813 tests**, all four gates green. 5 commits.

*(The previous pass — the v0.9.6 number-accuracy audit, P1–P6/A/B/C/D — is superseded by
this document; a copy is in the session scratchpad. None of its findings regressed.)*

---

## Scope of the sweep

| Agent | Scope | Findings |
|---|---|---|
| A1 | fresh-eyes full read of `collector.py` (1836 lines, line by line) | 6 |
| A2 | fresh-eyes full read of `tui.py` (4539 lines, line by line) | 6 |
| A3 | GPU + interconnect telemetry, verified with a fake pynvml | 6 |
| A4 | **mutation sweep** — 34 deliberate reversions vs the suite | 4 gaps |
| A5 | omissions + guard asymmetry (field-diff: parse sites vs every surface) | 8 |
| A6 | fresh-eyes `slurm.py` + `pending.py`, cross-checked vs live Midway3 | 3 |
| A7 | CLI + JSON/CSV, every finding backed by real command output | 6 |
| A8 | CPU + memory accounting, verified against synthetic cgroup trees | 4 |
| A9 | async lifecycle + leaks, growth **measured** not projected | 4 |
| A10 | docs-vs-behaviour + cross-cluster portability | 6 |

**53 findings.** Five were found independently by more than one agent — the strongest
signal in the set:

| Cross-voted finding | Votes |
|---|---|
| cgroup-v2 `memory.max=max` → false OOM-critical | 3 (A1, A8, my own trace) |
| CPU baseline clobbered → 0% frames *and* a latched 982-core peak | 3 (A1, A8, my own trace) |
| per-device `0.0` fabric-rate placeholder | 2 (A3, A5) |
| `gpu_monitoring_available` missing from CSV | 2 (A5, A10) |
| a frozen dashboard presented as live | 2 (A2, A9 — different mechanisms) |

---

## FIXED and shipped

### working tree — every gauge claimed 100% from 99.5 up, on both surfaces

`:.0f` reaches `100` at 99.5. Measured on the dashboard renderer:

```
99.40 -> 'used    ███████▉  99%'
99.60 -> 'used    ████████ 100%'      <- solid bar AND a boundary claim
99.99 -> 'used    ████████ 100%'
```

so the CPU, memory, GPU-compute and VRAM gauges, and the plain report's `Memory peak x / y (n%)`,
all said "at your limit" to a job with headroom. **This module had already fixed it one gauge
over**: `_time_frac_text` bounds the elapsed figure to `>99%` and its docstring gives both the
reason ("the two halves of that one line contradicted each other") and the family's spelling
(`rapidu.fmt.ratio_x` → `<0.01x`, `nodetop.core.duration` → `<1m`). The other gauges never got it,
and the memory one is what the off-node OOM guard fires on.

`units.pct_text` is the shared spelling, in `units` rather than either renderer because BOTH draw
this figure — the same reason `mem_pair` lives there, recorded at its own call site after a
`--mem=400M` job read `0.0 GiB / 0.4 GiB` in a renderer holding its own copy of the arithmetic.

**Both bar paths had to move with the label.** Each guard's comment says a solid bar must agree
with a `100%` label, and each keyed on `round(percent) < 100` — so in the 99.5–99.99 band the bar
and the label claimed 100% *together*, consistently and wrongly. Both now key on the unrounded
value; a `>99%` label is drawn with the last cell (unicode) or last character (ASCII) held back.

Only the top boundary moves: `0%` for a sub-0.5% value is deliberate (the bar beside it is empty,
and that is what the underuse advisory already says), values at or above 100 are untouched, and a
clock-skewed negative still reads `-3%` as `test_remote.py` pins.

`tests/test_gauge_percent_boundary.py` (40 tests). Teeth: reverting `pct_text` reddens 12,
reverting the two bar guards reddens 12. **Fourteen controls, all re-run green in both neutered
states**, covering every other value, the negative, the sub-0.5% empty bar, the 1% sliver, NaN,
zero width, and `_time_frac_text` keeping its own stronger rule (it keys on the time LEFT, so a
job whose budget really is spent still reads `100%`).


### working tree — D7: "FITS NOW" tested memory against configured, not free, capacity

The CPU half of the WHERE table has used idle `%C` since M5; the memory half used the aggregate
`sinfo %m`, which is a node's CONFIGURED size. So a partition whose nodes were all busy still
advertised their full memory, and a blank blocker is what puts the copy-pasteable requeue command
on screen.

**Reproduced end to end on the live cluster before anything was touched.** `vitelli-amd` holds 2
nodes, neither idle, and its largest mix node had **6,160 MB free of 250,000 MB configured**:

```
request    5 GiB/node -> fit_blocker = ''
request  100 GiB/node -> fit_blocker = ''
request  240 GiB/node -> fit_blocker = ''      <- "plausibly fits"
```

after:

```
request    5 GiB/node -> ''
request  100 GiB/node -> 'no room'
request  240 GiB/node -> 'no room'
```

`no room`, not `node too small`: the hardware is 250 GB and merely occupied, and `_shape_blocker`
draws that line against `max_config_node_mem_bytes`. Also verified on `caslake` (158,914 MB free of
193,274 configured): 140 GiB/node fits, 180 GiB is `no room`.

**Both halves of this row's why-deferred column were wrong, and measuring is what showed it:**

* *"Needs a new `sinfo -O AllocMem,MemSpecLimit` query"* — no new query. Adding **no** call,
  `Memory` and `AllocMem` ride along on `_fetch_free_gpus_by_partition`, which already visits every
  node line for `GresUsed`.
* **`MemSpecLimit` is not a valid `-O` field on this controller** (Slurm 20.11.8): `sinfo: error:
  Invalid job format specification: MemSpecLimit` on stderr, **rc=0**, and the column renders
  EMPTY — exactly the unsupported-field trap the `GresUsed` comment in that same function
  documents. Asking for it would have produced a silent column of blanks. It is moot regardless:
  **0 of 1,248** node lines on this cluster report one.

Two ordering details are load-bearing and pinned: the memory read happens BEFORE the GPU-specific
`continue`s, because those skip every node without GPUs and such nodes still have memory (464 of
1,248 node lines here have GPUs, and none of the CPU partitions where this was found); and
`isdigit` is what separates "0 MB free" (renders `0`) from "this Slurm does not know the field"
(renders ``), which is what makes `mem_detail` mean *unknown* rather than *full*.

`tests/test_pending_free_memory.py` (20 tests). Teeth: reverting `fit_blocker` to the configured
figure reddens 4; moving the memory read after the GPU skips reddens 2. **Eight controls, all
re-run green in both neutered states**, including that an unreadable field falls back to the old
behaviour rather than refusing a job on missing data.


### working tree — D5: the history deque was sized by an unclamped quotient

`history_seconds` has a ceiling (`MAX_HISTORY_SECONDS`, one day) and `poll_interval` has a floor
(`MIN_INTERVAL`, 0.1 s). Their QUOTIENT, which is what `deque(maxlen=…)` receives, had neither.

**Re-measured before anything was touched**, because nine items in this family have been found
already fixed while their row still called them open. This one was live, and the mechanism holds
— but **the row's own arithmetic was 2x high** and its verification line was taken at a
non-default `history_seconds`. What the numbers actually are, measured on this machine:

| | asked | interval (clamped) | slots per series |
| --- | --- | --- | --- |
| the blessed maximum | 86,400 s | 0.1 s | **864,000** (row said 1,728,000) |
| an hour at a second | 3,600 s | 1.0 s | 3,600 |
| half an hour at two | 1,800 s | 2.0 s | 900 |

At 864,000 slots across the 18-series shape: **596 MiB resident** on the node being monitored (the
row's "≈ 1 GB" is the doubled figure), and `_trend_tag` — `list()` + `min` + `max` over the whole
deque — cost **18.7 ms per call**, called twice a frame at ten frames a second, i.e. **37% of the
event loop**. Capped at `MAX_HISTORY_SAMPLES = 3,600`: **0.09 ms** per call.

Both halves the row's why-deferred column called for are in: the clamp
(`DashboardScreen._history_maxlen`) and the reported depth
(`DashboardScreen._history_window_seconds`), which every surface naming the window now reads, so a
capped window no longer advertises the depth it was asked for. The cap is on SLOTS, so a
reasonable configuration is untouched — verified above at 3,600 s/1 s, 1,800 s/2 s and the 60 s
default at the interval floor.

`tests/test_history_window_is_bounded.py` (41 collected). Teeth: dropping the clamp reddens 16,
reverting the reported window to the request reddens 6. **22 controls, all re-run green in both
neutered states.**


### working tree — pending: `sacctmgr` was re-run 8 times a second from inside `render()`

`PendingView._where` built the requeue tip from `resolve_user_associations(job.username or "")`
— **called on every render**. The panel is one `Static` and `PendingScreen._tick_spinner` is
armed with `set_interval(0.12, ...)`, so `render()` runs ~8 Hz for as long as the scheduler has
no start estimate for the job. Measured on a 110x40 headless screen, real timer, real 1 s window:

| 1.00 s in the spinner state | spinner ticks | `render()` calls | `sacctmgr` spawns |
| --- | --- | --- | --- |
| before | 8 | 8 | **8** (8.0/s) |
| after | 8 | 8 | **0** (0.0/s) |

Eight spawns a second happens only when the lookup FAILS — and that is deliberate.
`pending.resolve_user_associations` caches a success for the life of the process (account
configuration does not change while a dashboard is open) but returns `None` **without caching**
on error, so one bad `sacctmgr` cannot latch "unknown" over a whole session. Correct policy,
wrong place to pay for it: the retry was landing once per frame, on the event loop, with each
attempt allowed `SLURM_CMD_TIMEOUT` (15 s). With a `sacctmgr` that takes 200 ms to fail:

| 1 s window, 200 ms-to-fail `sacctmgr` | blocked inside `render()` | spinner ticks |
| --- | --- | --- |
| before | **1.00 s of 1.14 s** (88% of the loop) | 5 |
| after | **0.00 s** | 8 (full rate) |

**Fixed by moving the resolve, not by caching the failure** — the second of the two shapes the
deferred row named, because a short-lived negative cache bounds the *rate* of an event-loop stall
but not its *duration*: `render()` runs on the loop, so even one 15 s attempt per TTL freezes the
UI, and the first paint would still pay ~222 ms synchronously. `PendingView.assoc` is now an
input the poll fills in, exactly like `partitions` and `queue_rank`, and
`PendingScreen._refresh_once` resolves it **in the executor** as a fourth sibling of the
`asyncio.gather` that already runs `sinfo`/`squeue`/priority there. That also restores the
class's own documented contract ("a pure render over the resolved data"), which the resolver call
had quietly broken.

**Staleness window: one poll cycle (10 s), or sooner on `r`.** That is the right window because
it is the freshness every other figure on this panel already has — free nodes, idle cores, queue
depth, queue rank — so the association-filtered tip is not the odd one out, and a `slurmdbd` that
comes back is reflected on the next poll instead of being remembered as broken. `pending.py`'s
no-cache-on-failure policy is **untouched**: a failure is still not cached, it is simply retried
once per poll instead of once per frame. Before the first poll lands, `assoc` is `None` =
UNKNOWN, which costs nothing: `_where` returns above the tip while `partitions` is still empty.

`tests/test_pending_assoc_off_render.py` (13 tests). Teeth: neutering all four sites (the
attribute, its import, the `_where` read, and the poll's gather task/assignment/fallback) reddens
6 of the 13 — the timer test back to 4 spawns in 0.5 s, the 5-tick test to 5 spawns, a
booby-trapped resolver back to being called from `render()`, and the three poll tests to
`'PendingView' object has no attribute 'assoc'`. Seven controls, verified green in BOTH states:
the tip still names the QOS from the table, an unknown table still softens the command to a bare
`scontrol update`, a partition the user cannot submit to is still dropped from the tip, the
loading state still needs no table at all, the poll is still every 10 s, the panel still says the
same things — and `pending.py` still refuses to cache a failure while caching a success once. The
two existing tests that wired the table through the module-level resolver
(`test_pending.py::TestPartitionMoveCommandIsComplete`, `test_requeue_hedge_ascii.py`) now set
both that and `view.assoc`, so they too pass in both states (32 passed either way).

### working tree — the pending view repainted its whole panel 8 times a second

Reported from use: in the `Where It Could Run` table, "all these YES are flickering". The
verdicts were neither wrong nor animated — they were being **re-sent to the terminal on every
spinner frame**.

`PendingView` is ONE `Static`: its `render()` returns why the job waits, when it might start, and
the WHERE table as a single string. `PendingScreen._tick_spinner` is armed with
`set_interval(0.12, ...)` (8.3 Hz) and ended in a bare `view.refresh()`. A bare refresh dirties
the widget's whole region, and Textual's partial update turns a dirty region into spans and
re-emits **every line in it without comparing content** — so animating one braille glyph rewrote
the table, verdict cells included.

Measured on a 110x40 terminal with a pending job the scheduler had not yet planned (the only
state the spinner runs in):

| | 8 spinner ticks |
| --- | --- |
| before | 10 compositor updates, **31,430 bytes**, 10 distinct spinner glyphs |
| after | 10 compositor updates, **2,670 bytes**, 10 distinct spinner glyphs |

and, per tick, `Static.refresh` went from one whole-widget call to region calls of **height 1** at
exactly the line indices whose painted text changed. The WHERE section's painted text is
identical across consecutive frames — 0 of 5 differ — which is what made this a pure repaint
defect rather than a value defect.

**Not the layout-pass storm the picker was fixed for.** `Widget.refresh` defaults to
`layout=False`, so a spinner tick produced **zero** layout passes; the volume was the problem.
The one `layout=True` was the 10-second poll, which fired even when the poll resolved an
identical panel — that is the second half of this fix, and it is the guard
`JobSelectorScreen._tick` already states.

`tests/test_pending_spinner_repaint.py` (14 tests). Teeth: restoring the bare `view.refresh()`
reddens the whole-panel assertion; restoring the unconditional `layout=True` reddens the
identical-poll assertion. Five controls, verified green in the pre-fix state: the spinner still
advances every tick, a planned job still animates nothing at all, a changed poll still repaints,
the panel still says the same things, and a torn-down screen still does not raise.

### working tree — pending: a dead alias that kept its own import looking used

`tui.py` carried `_cpu_ratio = cpu_ratio` — **one occurrence in the whole package**, zero in the
tests. A leftover: `_cpu_health` moved its rule into `model.cpu_is_underused`, and the comment
six lines below the alias still says so ("The rule lives in model.cpu_is_underused so the
plain-text summary reaches the same verdict").

**It was invisible to both of the obvious checks.** `F401` sees nothing, because the alias keeps
the import *used*; and a zero-reader grep for `cpu_ratio(` finds the definition plus the one real
call in `model.py:138` and stops. Removed, along with the import it was holding up.

**Found by following a coverage gap to its caller.** `model.py:125` — `cpu_ratio`'s
zero-allocation guard — was uncovered, which raised the question of who calls `cpu_ratio` at all.
The answer: one real caller that short-circuits first (`cpu_is_underused` is
`cores_allocated > 1 and cpu_ratio(cpu) < threshold`), plus this alias. The guard itself is left
alone — unreachable through that caller and defensive by design — and a control now pins it so
the removal is not mistaken for permission to delete it.

`tests/test_no_dead_import_aliases.py`, 24 tests: an AST sweep over all 15 modules for a
module-level `alias = imported_name` with no `Load` of the alias anywhere, plus five detector
guards on planted sources (a live alias, a literal, an alias of a local definition, a nested
assignment) so the sweep cannot pass by scanning nothing.

Teeth: restoring the alias reddens exactly `[tui.py]`, 23 green.

### working tree — pending: `join_bounded` was the half of `aio` nothing tested

Found by mining **partial-branch coverage** rather than by reading — a first for this campaign.
`aio.py` is 14 statements and had exactly one uncovered: line 42, the `raise exc`. The module's
sibling has two direct tests (`test_cli.py::test_the_reap_does_not_eat_the_callers_cancellation`
and `test_it_still_reaps_the_future_it_cancelled`); `join_bounded` had **none**, despite four
call sites — two in `collector.teardown`, one in its sample path, one in `tui`'s poll-task join.
The asymmetry inside one 62-line module is the tell.

Two stated rules, both now pinned:

* **"A failure the awaited work reported is still surfaced, because teardown should not quietly
  eat a real error."** That is line 42.
* **Why the primitive is `asyncio.wait` and not `wait_for`:** *"wait_for CANCELS its awaitable on
  timeout and then waits for that cancellation to land, so a task which does not honour
  cancellation makes `wait_for(task, T)` wait forever — a bound in name only."* Pinned with a
  task that genuinely ignores cancellation, which is the only shape that tells the two
  primitives apart.

`tests/test_bounded_join.py`, 8 tests, no cluster and no fixture — pure asyncio. Teeth, two
neuters:

| neuter | red |
| --- | --- |
| the `raise exc` removed | the 2 reported-failure tests |
| `asyncio.wait` swapped for `wait_for` | the 2 bound tests **and** the cancelled-future control |

That third failure is the interesting one and it is why the control is labelled as belonging to
both halves: with `wait_for`, awaiting an already-cancelled future raises `CancelledError` at the
caller, so the `if not finished.cancelled()` guard stops being enough. Running the neuters is
what established which test guards which half.

**Every test carries its own `@pytest.mark.timeout(10)`**, because the failure mode under test IS
a hang: the `wait_for` neuter took 20s to report even with the tight bounds, and without them it
would have sat until the suite-wide 120s cap saying nothing.

### working tree — pending: `--interval`'s help names four numbers, all of them copies

    Polling interval in seconds (default: 0.5 for TUI, 1.0 for headless;
    raised to a 0.1s floor on the node, 1.0s off it)

Each is a copy of a value that lives in `config.py`: `0.5` is
`SlurmwatchConfig().poll_interval`, `1.0` is `.headless_interval`, `0.1` is
`MIN_INTERVAL`, `1.0` is `MIN_REMOTE_INTERVAL`. Nothing compared them, and this is the flag
most likely to be tuned — the interval is forwarded, floored and re-floored across three code
paths in this package.

Two sibling packages already run this check where their parsers are reachable
(`nodetop/tests/test_stated_defaults.py`, and `slurmpast` gained one the same day) and rapidu
sidesteps it entirely by interpolating `%(default)s`. Those compare `action.default` to a
`(default: X)` regex, which cannot work here: `--interval`'s own default is `None`, because the
two real values are chosen later by mode. So the comparison is against the CONFIG the modes
read, which is where the numbers actually are.

`tests/test_stated_interval_defaults.py`, 14 tests. Teeth, three drifts, each reddening exactly
the assertion for the number that moved with 13 green:

| neuter | red |
| --- | --- |
| `poll_interval` → 0.6, help unchanged | the TUI-interval assertion |
| `MIN_INTERVAL` → 0.2, help unchanged | the on-node-floor assertion |
| help → "0.3s floor", constants unchanged | the on-node-floor assertion |

The last two reddening the same assertion is the point: it compares the pair rather than
trusting either side.

**Two things the file records because they cost a wrong first attempt.** The expected strings
are generated from the constants, and the format spec has to match the help's own spelling —
`:.1f`, not `:g`, because `f"{1.0:g}"` is `"1"` while the sentence says `"1.0"`. And
`headless_interval` and `MIN_REMOTE_INTERVAL` are **both 1.0**, so each assertion anchors on the
words around the number ("for headless", "off it") rather than on the number alone; otherwise
one could move and the test would still find a `1.0` in the sentence.

### working tree — pending: the Python floor was a rule stated in prose with nothing reading it

`pyproject.toml` declares `requires-python = ">=3.10"`, `ci.yml` runs a 3.10 job, and this repo
already states the rule that follows — in prose, at `test_forward_compat_metadata.py`'s
`_pyproject()`:

> Read as text, deliberately. `tomllib` is 3.11+ and 3.10 is the oldest version this file
> asserts support for, so importing it would make the test unrunnable on the interpreter it
> most needs to run on. The siblings' equivalents record the same reason, and `release.yml`
> avoids tomllib for it too.

**Nothing checked it.** The convention was followed by hand in that one file and nowhere else —
the shape this loop keeps finding, a rule the code states about itself that no test reads. It
was found the only way it could be: by breaking it. A test written this round imported
`tomllib` to read the `[tool.ruff.lint]` table, and all four slurmwatch gates passed it, because
a local run is one interpreter and that interpreter is 3.11. **slurmpast's equivalent caught the
identical import immediately** — and its own docstring records the same defect costing a CI
round-trip: "Four of its assertions died with `ModuleNotFoundError` in CI's py3.10 and
oldest-Textual jobs, having passed every local gate."

`tests/test_python_floor_is_enforced.py` ports slurmpast's sweep, deliberately keeping its table
and its `try:`-guarded exemption so a fix in one transfers — the same argument
`test_forward_compat_metadata.py`'s `BANNED` list already makes about itself ("Same list
slurmate and slurmpast scan for, so a fix in one transfers"). `sys.stdlib_module_names` cannot
answer the question, being the *running* interpreter's, so the four-entry table is the
alternative; a module missing from it is one CI round-trip, not a false pass.

Teeth: planting `import tomllib` unguarded in a test file reddens the sweep and nothing else
(1 red, all 3 controls green, including the one that keeps a `try:`-guarded backfill legal).

**The offending import is gone from both new files this round** — the ruff config values each
live on a single `pyproject.toml` line, so a regex reads them and no parser is needed.

**Family survey while here:** nodetop enforces this in `test_readme.py`, rapidu in
`test_py36_compat.py` (floor 3.6), slurmpast in `test_portability.py`, slurmwatch now here.
**slurmate has none and declares the same `>=3.10`** — the remaining gap, and its
`test_forward_compat_metadata` sibling already shares the `BANNED` list, so the port is the same
shape again.

### working tree — pending: two `noqa` directives claimed a violation the line does not have

The suppressions were the one claim in this repo nothing checked. Two had gone stale:
`tests/test_collector.py`'s `# noqa: F811` on a fixture ruff no longer flags, and a
`# noqa: E402` on an import following a `sys.path.insert`. Both codes are in `select` and
neither is exempted for `tests/*` (that entry is `ARG`, `N802`, `N815`), so nothing was
suppressing them — each line advertised a violation it does not have.

**Both were BARE**, the code with no rationale after it, and that is what made removing them
right. The same sweep across the family found stale directives in `nodetop` (2) and `rapidu`
(5) where every one carries a reviewer's note — `# noqa: S603 - fixed argv, never a shell`,
`# noqa: BLE001  (a hang is worse than a report)` — and this rule would demand deleting the
note to satisfy the linter. Not enabled there, deliberately.

**Measured on both ends of the dev bound first**, because a directive one ruff calls unused can
be load-bearing on another and CI resolves the upper end: 0.15.18 and 0.16.6 agree on both, and
on every repo's count. The two directives that REMAIN in `test_collector.py` are now proven
load-bearing by the same rule.

`RUF100` is in `select`, so the gate keeps them honest and nothing in `tests/` re-implements
`ruff check`. Teeth: dropping it from `select` reddens 2 and the planted stale directive goes
from 1 finding to 0; all five controls green.
`tests/test_stale_suppressions_are_caught.py`, 8 tests.

**Verified CLEAN alongside it:** no stale `# type: ignore` anywhere in the family.
`mypy --warn-unused-ignores` against each repo's own scope reports **Success** for slurmate
(src/), slurmpast and slurmwatch (src/ tests/) — all 24 ignores across the three are
load-bearing. nodetop and rapidu already set `warn_unused_ignores` in `pyproject.toml`, so
theirs were policed by their own gate all along; slurmate, slurmpast and slurmwatch do not,
which is why they were the ones worth probing.

### working tree — pending: two figures on the demo node card never moved

`_vary_mock_for_node` exists for exactly one reason and states it: the per-node mock frames
"were byte-identical apart from the hostname: every node read 50.0% CPU and 25.0% memory ...
pressing the key changed nothing on screen. That is indistinguishable from a switch that
silently failed, which is the exact confusion the switch banner exists to prevent."

Two fields survived that fix. It scales `current_bytes`, `working_set_bytes`, both memory
percentages, the whole CPU triple **including `peak_effective_cores`**, and every GPU figure
— and left `peak_bytes` and `peak_working_set_bytes` alone. Measured on the mock's own frame:

| node | mem_used | mem_peak | ws | ws_peak |
| --- | --- | --- | --- | --- |
| 0 | 16.00 | 16.80 | 16.00 | 16.80 |
| 1 | 13.28 | **16.80** | 13.28 | **16.80** |
| 4 | 5.12 | **16.80** | 5.12 | **16.80** |

So node 4 rendered `used 5.12 GiB` beside `peak 16.80 GiB` — a 3.3x ratio where
`_collect_memory`'s mock branch generates 1.05x — and both peaks read the same bytes on all
five nodes, which is the original defect surviving on the two fields the fix missed.

Both peaks are now scaled by the same factor and **floored at the scaled reading**, because
`peak >= used` is the invariant `_apply_peaks` states and every producer upholds. The floor is
load-bearing, not decorative: at factor 0.32 `int()` truncation pulls a peak only 1.05x above
its reading down onto or under it, which a dedicated test asserts by driving `peak_ratio=1.0`.

Demo-only (`collector.py:846`, "Demo only: make a switch visible"), so this is polish on the
showcase rather than a telemetry correction — but the showcase is where anyone tries the node
switcher without a multi-node allocation, which is the whole argument the docstring makes.

Teeth: removing both lines reddens **6**, all **19** controls green. One of those controls is
the invariant itself, and it is a control for a reason worth recording — leaving the peaks
UNSCALED left them above every scaled reading, so `peak >= used` survived the defect. It
guards the other direction, that scaling did not pull one under its own reading.
`tests/test_mock_node_variation.py`, 25 tests.

**Verified CLEAN alongside it, do not re-probe:**
* **The plain summary against `--json`.** `_print_remote_summary` vs `snapshot.to_json()` on
  three snapshot shapes (typical, no-limit, sub-KiB): every rendered integer is reachable once
  GiB divisions and `H:MM:SS` splits are credited. slurmwatch was the last repo this sweep had
  not covered.
* **That summary's `--ascii` fold**, measured both ways so it is not vacuous: one em dash
  without the flag and zero with it, on all three shapes.
* **The `peak >= current` invariant on every producer.** Four build a `MemoryMetrics` and each
  upholds it by a different mechanism: off-node sets `working_set_bytes` and
  `peak_working_set_bytes` to the same `rss`; the mock generates `min(int(1.05 * current),
  limit)` and `pct` caps at 72 so the `min` never bites; the cgroup path feeds
  `_advance_working_set_peak` the same `working_set_bytes or current_bytes` expression it
  publishes; and `_apply_peaks` floors `peak_bytes` explicitly because the kernel counter is
  the one source that can fall. A synthetic probe that forces a violation is measuring an
  unreachable state — checked before believing it. Only ONE of the four was asserted, and
  incidentally: `test_collector.py:1382-1383` sits inside
  `test_remote_cpu_peak_is_populated_not_zero`, whose subject is the CPU peak not being zero.
  The comment at `collector.py:798` ("Both paths") means on-node/off-node and is correct — it
  is not overclaiming, and was left alone.

### working tree — pending: half the underuse sentence could still drift

`tui.py` has said this at the advisory's call site for several rounds:

    # The tail comes from model.CPU_UNDERUSE_ADVICE so this line and the
    # plain-text summary's cannot drift; only the ink on the flag differs.

True of the tail. Not true of *this line*. The subject half --
`only ~N of M cores are doing work` -- was spelled once in `cli.py:1442` and once in
`tui.py:2859`, and the subject is the half that actually drifted: `format_cores`'s own
docstring records the incident ("the card said `only ~1 of 8 cores are doing work` and the
plain summary said `only ~1.0 of 8 ...`"). The previous round fixed that by sharing the
**formatter**, which repaired the instance and left the sentence in two homes.

Found by intersecting every module pair's string literals -- the same probe that found the
figure. Of 15 modules, all pairs, exactly two prose literals were shared: this one and
`' has no attribute '` (`__init__`/`_version` boilerplate).

`model.cpu_underuse_subject` is now the one home, beside the `CPU_UNDERUSE_ADVICE` it
precedes, so the comment describes what the code does. Markup stays with the callers --
the helper returns words only, asserted, because a Rich tag in the shared clause would
leak into the plain summary.

Teeth, per surface, because the fix has two call sites and the two failure modes are
different. Re-spelling one site *identically* changes no output, so only the source pin
catches it; re-spelling it as a drift is what the behavioural half is for:

| neuter | re-spelled identically | re-spelled `cores are busy` |
| --- | --- | --- |
| `cli` only | 1 red (source pin) | 6 red (plain + cross-surface) |
| `tui` only | 1 red (source pin) | 6 red (dashboard + cross-surface) |

All four `TestControls` green in all four runs. `tests/test_underuse_sentence_single_source.py`,
19 tests, reusing `test_core_figure_single_source.py`'s harness and its measured gating
(the advisory fires strictly below `0.15 * cores`, and `cli` refuses it off-node).

**Also verified clean by the same probe, and not to be re-run:** all module pairs in
`slurmpast` (19 modules) and `rapidu` (11) share no prose literal at all.

### working tree — pending: a fractional-GPU job was told it had asked for no GPU (D18)

**"no GPUs requested by this job", about a job holding `gres/shard:2`. (LOW)**
`_parse_tres_gpus` matches `gres/gpu[:type]=N` and `_parse_gpu_count` matches
`[gres/]gpu[:type]:N`; neither has ever matched Slurm's two SHARED-GPU GRES,
`gres/shard` and `gres/mps`. Off the node there is no NVML and no
`CUDA_VISIBLE_DEVICES` to fall back on, so the request collapsed to
`gpu_count_requested=0` and both dashboard sites read that zero as "none". Measured
from one login-node `scontrol` record (`TRES=...,gres/shard=2`,
`TresPerNode=gres:shard:2`, `GRES=shard:2(IDX:0)`), rendered through the same
`_dashboard_surfaces` / `_print_remote_summary` pair the D17 round used:

```
  ● GPU     none requested                       <- dashboard row
  no GPUs requested by this job                  <- the `g` drill-in headline
  (no GPU line at all)                           <- the plain report
```

Control on the same record shape: `gres/gpu=2` → `gpu_count_requested=2`. Checked
against the committed state as well, since six items in this family turned out already
fixed: `git show HEAD:` has no `shard`/`mps` handling in `slurm.py`, `model.py`,
`tui.py` or `cli.py`, and HEAD carries both claims verbatim (`tui.py:1957`,
`tui.py:2848`). **The row's "self-heals on the node" half is corrected**: healing runs
entirely through a job process's `CUDA_VISIBLE_DEVICES` (`_resolve_gpu_indices` →
`ctx.gpu_count_requested = len(gpu_indices)`), measured 1 device with it exported and
still **0** with the environ unreadable, and the exact-index path contributes nothing
because `_GRES_IDX_RE` requires a literal `gpu` and returns `[]` on
`GRES=shard:2(IDX:0)`.

**Shape: withhold the negative, do NOT invent a count.** Counting shards would have
traded a false zero for a false device count — two shards can be two slices of ONE
physical GPU and `mps:100` is 100% of one — and `gpu_count_requested` is what the
collector then goes looking for devices with (`collector.py:566`). So
`slurm.py:1694 _parse_gpu_fraction_request(record)` carries the REQUEST, in Slurm's own
words, on a new `JobContext.gpu_fraction_request` (`model.py:965`);
`gpu_count_requested` deliberately stays 0 and is asserted to. Per-node fields
(`TresPerNode`/`Gres`) are read before the job-wide `TRES`, so the field means what
`gpu_count_requested` means — this node's request. After:

```
  ● GPU     shard:2 requested — a fraction of a device, not a whole GPU
  shard:2 requested — shard/mps allocates a fraction of a device, not whole GPUs, so
  there is no device count to report. Run on the compute node for live GPU utilization.
  GPU      shard:2 requested — a fraction of a device, not a whole GPU;
           run slurmwatch on the compute node for live GPU utilization
```

All three sites, because "fixed on one side only" is this file's recurring defect:
`tui.py:2035` (the `ResourceRows` GPU row, which now takes `job_ctx` beside the
snapshot it already had — the request is a static scontrol fact, not a measurement),
`tui.py:3052` (the drill-in headline, where there is room to say WHY there is no count,
and which withholds the "run on the node" advice when the reader is already there), and
`cli.py:1460` (`_print_remote_summary`, which said nothing at all). No new Slurm query:
the GRES was in the record already. Both dashes are `ascii_mode`-folded. The machine
payloads are untouched and still publish `gpu_count_requested: 0` for such a job — the
snapshot schema is a separate contract with its own round-trip gates, and this round did
not widen into it.

`tests/test_gpu_fraction_request.py`: 9 tests that read the fix (3 on the parse, 4 on
the dashboard's two sites, 2 on the report) and 2 controls. **Teeth, five ways** —
neutering the parse helper's two `return f"{m.group(1)}:{m.group(2)}"` reddens exactly
the 3 parse tests; neutering only its call site in `resolve_job_context` reddens 2 of
those (the record-driven pair) and leaves the direct helper test green; neutering only
the `tui.py` row branch reddens the 2 row tests; only the drill-in branch, the 2
drill-in tests; only the `cli.py` branch, the 2 report tests. All five out at once: 9
red / 2 green, so both controls were run and passed with the fix IN and OUT — and
neither shares an input with a finding (one job asks for whole devices, the other for no
GPU at all).


### working tree — pending: nothing held the source to the interpreter range it advertises

**No guard against APIs a newer Python removes. (LOW)**
`requires-python = ">=3.10"` has no upper bound, so pip installs this on whatever
comes next. `src/slurmwatch` uses none of `distutils`, `import imp`, `utcnow`,
`getdefaultlocale`, `find_loader`, `pkg_resources` or `typing.ByteString` — but only
because nobody had reached for one. slurmate has scanned for exactly that list since
the round its CHANGELOG describes, and slurmpast gained the same scan in its round
sixty-five; slurmwatch had neither half.
`tests/test_forward_compat_metadata.py` adds it, with the planted-offender control the
siblings use, because a guard that cannot fail is not a guard.

**And the advertised versions are now pinned to the tested ones. (LOW)**
The classifiers list 3.10–3.13 and CI runs exactly 3.10–3.13. That agreement was
unchecked, so either list could have moved alone: a dropped classifier narrows what
PyPI shows for a version that is actually tested, and an added one promises a version
nothing runs.

**Deliberately NOT done here: the 3.14 classifier.** slurmpast added one because its
own `issues.md` records the published artefact installed from PyPI onto midway2 at
Python 3.14.6 and exercised against real accounting history. **slurmwatch has no such
run recorded anywhere** — measured, not assumed: `3.14` appears in no README,
CHANGELOG or issues entry here. A classifier is a promise to installers, so the
invariant pinned is the one that is true today, "advertised == tested". If a 3.14 run
is ever recorded, `test_the_classifiers_are_exactly_the_tested_versions` is what will
fail, which makes adding the claim deliberate rather than drift.

Both halves read `pyproject.toml` as text rather than through `tomllib`, for the reason
the siblings record: tomllib is 3.11+ and 3.10 is the floor these tests assert, so
importing it would make them unrunnable on the interpreter they most need to run on.


### working tree — pending: ALL FOUR mypy-excluded test files now pass CI's real scope

**The local gate had been weaker than CI's for many rounds. (MED — process, not runtime)**
`[tool.mypy]` carries **no** excludes and both `ci.yml` and `release.yml` run a bare
`mypy src/ tests/`, but the polish loop had been invoking it with four `--exclude`
flags. Run as CI runs it, the tree held **184 errors** across those four untracked
files: 100 `no-untyped-def`, 59 `no-untyped-call` cascading from them, 23
`attr-defined`, 2 other. Every "gates green" report for those rounds was measured
against a gate that does not exist.

All four are now annotated and clean, so **the exclude list is empty**: `mypy src/
tests/` reports success over 36 source files with no flags, alongside `ruff check .`,
`ruff format --check .`, and `pytest --cov=slurmwatch --cov-fail-under=60` at **2157
passed, 93.06%**. slurmwatch passes its real CI gate end to end.

`test_portability_round88.py` held the 23 `attr-defined`, in two patterns, and both
fixes are behaviour-identical rather than suppressions:

* **`slurm.subprocess` IS the stdlib module object** — `slurm.py` does a plain `import
  subprocess`, and `slurm.subprocess is subprocess` returns True. So importing
  `subprocess` in the test and patching it patches the very same attribute. 21 sites
  changed; the file's 75 tests still pass.
* **`SlurmCommandError` lives in `slurmwatch/exceptions.py`** and `slurm.py` only
  re-imports it, which `strict = true` (`no_implicit_reexport`) refuses. Imported from
  `exceptions` instead, and a redundant local re-import dropped.

Two further things were not annotation noise but real errors mypy surfaced once the
file was in scope — which is the argument against excluding it in the first place:

* `_no_name_service` returns the recorded `squeue` calls, and three tests iterate that
  return. I first typed it `-> None`; mypy reported those callers iterating `None`.
* the two pty-drive helpers return `code` and `signalled` that are **Optional**. A
  `dict[str, int]` looked right (`bool` subclasses `int`) and hid it. Now two
  `TypedDict`s, which keep `closed`/`opened` as `int` so `got["closed"] >=
  got["opened"]` stays type-checked rather than becoming a comparison on `object`.

Also fixed while there: a `lambda *a, **k: asked.append(a) or _Result(...)` — correct,
since `append` returns `None`, but it reads as a value-returning call and mypy flagged
it. Replaced with a named recorder that appends and returns.

### working tree — pending: the release gate's Python endpoints were unpinned

**`release.yml` must track `ci.yml`'s floor, and nothing checked it. (LOW)**
`release.yml` runs the suite itself rather than trusting `ci.yml`, because "a tag push
went straight to build-and-publish with no dependency on the suite at all -- and a PyPI
upload cannot be taken back". It gates on the **endpoints** of `ci.yml`'s matrix and
runs `pytest -q` without the coverage flags.

**Both differences are deliberate and are NOT changed here.** I first read them as drift
from that comment, then measured all five sibling packages and found the identical shape
5/5 — which is a convention, not an accident. Widening those matrices would have been
redesigning the release policy of five packages on the irreversible path.

What is genuinely invariant and was unchecked: the release matrix equals `{min, max}` of
`ci.yml`'s (true in all five, including rapidu's 3.9 floor), the lint/type commands are
spelled identically on both paths, and `requires-python` does not promise a version
below the gated floor. `tests/test_release_gate_tracks_ci.py` pins those three and
deliberately leaves the pytest invocation and the middle of the matrix alone. Teeth
confirmed against the real hazard: raising `ci.yml`'s floor without moving `release.yml`
reddens it.


### working tree — pending: dead `Footer` CSS, a floor comment crediting unused widgets (D19)

**Ten lines of CSS styled widgets the app never mounts. (LOW)**
The dashboard's CSS carried `Footer` and `FooterKey` rules, but this package imports
neither: the keybinding bar is its own `KeyFooter(Static)`, "precisely because the
stock Footer paints every key the same accent". Textual does not warn about a selector
that matches nothing, so it read as live styling for a widget that was not there.
Measured before removing: **no `Footer(` anywhere in `src/` or `tests/`**, and of the
11 widget-class selectors in the app's CSS exactly those 2 were unbacked by an import
or a local class.

**The `textual>=0.86` floor comment credited widgets nothing imports. (LOW)**
It claimed the floor "covers every widget the dashboard uses (Sparkline, DataTable,
Digits, Rule, Header/Footer)". Actually imported: `Digits, Header, ListItem, ListView,
RichLog, Static`. `Sparkline`, `DataTable` and `Rule` appear nowhere — the only "Rule"
in `src/` is the word in a comment and a `Static` with `id="selector-rule"`. The real
reason for the floor was already the comment's first sentence: the theme system.
Reworded to the real imports, without re-listing the removed names — the history lives
here instead, which is what let the pin below stay simple.

`tests/test_css_selectors_are_mountable.py` pins both: every widget-class selector in
the CSS is a class this module can mount, and the dependency comment credits no widget
the package does not import.

### working tree — pending: the `~` collapse had no test for its boundary (replaces D21)

**Nothing checked that the home collapse stops at a path COMPONENT. (LOW)**
`_shorten_path` guards with `path.startswith(home + os.sep)`. Drop the `+ os.sep` and
`/home/alicex/work/run.sh` renders as `~x/work/run.sh` — another user's path relabelled
as the reader's own home, in the field a reader uses to confirm whose job they are
looking at. **Demonstrated: under that neuter all five pre-existing `TestShortenPath`
tests still pass**, which is what made the gap invisible.
`tests/test_home_collapse_boundary.py` adds the negative (three prefix-only paths, a
shorter sibling) and sets `HOME` explicitly, so the assertions measure the rule rather
than the machine. D21's own claim is withdrawn above with the values that disprove it.


### working tree — pending: the requeue hedge kept its em dash under `--ascii`

**The plain report's own promise, broken one line below where it is made. (LOW)**
`_print_pending_summary` opens with `# Honour --ascii here too (a non-UTF-8 terminal /
pipe): no stray Unicode.` It folds its own `dash`/`dot`/`dots` and passes `ascii_mode`
into `explain_reason` — then printed `partition_move_caveat` raw, which hardcoded U+2014.
The two lines print together, so `--ascii` produced a report contradicting itself:

```
  Tip    broadwl has room for this request now - requeue with: scontrol update JobId=777 ...
         add QOS=<name> if that partition needs its own — the QOS does not move with it
```

The fold now lives in the producer, exactly where :func:`explain_reason` already keeps
its own. **Why the plain report and not the dashboard:** `PendingView.render` ends in
`_asciify(out) if ascii_mode else out`, so it folds everything it has built and never
leaked; the plain report folds token by token and helper by helper, so one unfolded
string reaches the screen intact. Measured rather than assumed — handing the view an
unfolded caveat still renders ASCII-clean — so the dashboard's call site deliberately
does **not** pass the flag, and a control pins that asymmetry.

Swept for siblings before fixing: `partition_move_caveat` was the only text helper in
`pending.py` carrying non-ASCII with no way to fold it. `_explain_reason` also does, but
it is private and reached only through the folding `explain_reason`.

**Why nothing caught it:** the one `ascii_mode=True` test of the plain report stubs
`resolve_cluster_partitions` to `[]`, so the Tip branch never runs, and the suite's only
`isascii()` assertion covers the dashboard's *loading* state.
`tests/test_requeue_hedge_ascii.py`.


### working tree — pending: the WHERE table's free capacity had no denominator (D17)

**Two partitions, one 94% free and one 7% free, rendered the SAME row. (LOW)**
`resolve_cluster_partitions` sums `total_nodes` and `cpus_total` from `sinfo` %D/%C —
and before this round **nothing in `src/` read either field**: measured with
`grep -rn '\.total_nodes\|\.cpus_total' src/`, the only hits were the two `+=` that
write them (`pending.py:785,787`). The table therefore printed a numerator alone, so a
job's two alternatives came out character-identical:

```
           partition         free nodes  idle cores   gpu            can run now?
           small-part                 6         240   —              FITS NOW
           big-part                   6         240   —              FITS NOW
```

Both rows are the same string after the name (asserted, not eyeballed), yet `small-part`
has 240 of 256 cores idle and `big-part` 240 of 3200. `resolve_cluster_partitions` sorts
by **absolute** free capacity (`-(idle_nodes + mix_nodes), -cpus_idle`), so the `Tip`'s
pick between two such rows had nothing on screen to justify it. Now:

```
           partition           free nodes     idle cores   gpu            can run now?
           small-part                 6/8        240/256   —              FITS NOW
           big-part                 6/100       240/3200   —              FITS NOW
```

One helper, `pending.py:1265 capacity_cell(free, total)`, called at both columns of
**both** renderers — `cli.py:2995-2997` (`_print_pending_summary`) and
`tui.py:5583-5588` (`PendingView.render`) — because "fixed on one side only" is this
file's recurring defect and this table has been an instance of it. No new query: the totals were already collected and thrown away.
`total` of 0 means unknown (a caller-built `PartitionResources`, or an `sinfo` with no
%D/%C) and then the bare figure still prints — "240/0" would claim more, not less.
The numeric columns widened 10→13 / 11→13 / 12→13 cells so a real cluster's
`10458/12800` cannot shove the verdict marker out of line; alignment across magnitudes
is pinned by a control at both widths.

`tests/test_where_denominator.py`: 7 denominator tests (report and card covered
separately) and 5 controls. **Teeth, three ways** — neutering
`capacity_cell` reddens 7 and keeps all 5 controls green; neutering only the `cli`
call sites reddens exactly the report-side 4; only the `tui` sites, the card-side 4.


### working tree — pending: the pending card never said WHICH QOS throttles the job (D16)

**Both other job cards carry a `qos` chip; the pending one dropped a field it already had. (LOW)**
`resolve_pending_job` parses `QOS=` off the same `scontrol show job` record as every other
`PendingJob` field (`pending.py:451`) and **nothing in `src/` read it**: measured with
`grep -rn '\.qos\b' src/slurmwatch/`, all 5 hits are `JobContext.qos` — `tui.py:2238`
(`JobDetailsPanel`, the running card), `tui.py:6028` (`ForeignJobView._provenance`, the
foreign card), plus `model.py:673` / `collector.py:858` carrying it into CSV and the
snapshot. `grep -n qos src/slurmwatch/cli.py` returned **zero lines**. Checked against the
committed state as well, since six items in this family turned out already fixed:
`git show HEAD:src/slurmwatch/pending.py | grep -n qos` gives the field at `:80` and the
write at `:426` and no read, and `git show HEAD:src/slurmwatch/tui.py` has the same two
`ctx.qos` chips. The one read anywhere was a parse assertion,
`tests/test_pending.py:134`.

Reachable, and reachable often: `explain_reason` funnels every `qos`-containing reason code
into one sentence (`pending.py:295`) — *"A QOS limit is capping your usage (jobs / CPUs /
GPUs / memory / time / billing)."* — so `QOSMaxJobsPerUserLimit`,
`QOSMaxWallDurationPerJobLimit` and `InvalidQOS` all named a QOS this view could identify
and would not print. Before / after, with `qos=gpu-throttle`:

```
  ● PENDING   ·   QOSMaxJobsPerUserLimit
  ● PENDING   ·   QOSMaxJobsPerUserLimit   ·   qos gpu-throttle

Job 777  gpu-shared  PENDING  name `train`
Job 777  gpu-shared  qos gpu-throttle  PENDING  name `train`
```

The same field on **both** renderers — `tui.py:5353-5357` (`PendingView._why`, in the
`_DIM`-label / `_GPU_COLOR`-value chip form the two other cards use) and `cli.py:2844-2845`
(`_print_pending_summary`'s identity line) — because "fixed on one side only" is this
file's recurring defect. No new Slurm query: the value was parsed and then thrown away. An
absent QOS (`_clean` folds `(null)`/`N/A` to `""`) prints no chip and no orphaned
separator, and the name is escaped before it reaches Textual's markup parser.

`tests/test_pending_qos_named.py`: 8 tests that read the fix (4 dashboard, 3 report, 1
cross-surface), 2 empty-QOS guards, 4 controls. **Teeth, per surface** — neutering only
`tui.py`'s `{code}{qos}` reddens the 4 dashboard tests plus the cross-surface one and
leaves the 3 report tests and all 4 controls green; neutering only `cli.py`'s
`{pending.partition}{qos}` reddens exactly the 3 report tests plus the cross-surface one.
Both out: 8 red / 6 green, so every control was run and passed with the fix IN and OUT.


### `bf9a089` — collector: NVLink scope, cgroup-v2 OOM basis, CPU baseline

**1. NVLink throughput measured ONE LINK, not the fabric. (HIGH)**
`NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX/TX` are *per-link* counters selected by
`nvmlFieldValue_t.scopeId`. pynvml populates `scopeId` only when the caller passes a
`(fieldId, scopeId)` **tuple**; a bare int leaves it at the ctypes default **0**. Every
reported rate was therefore link 0 alone — **1/18 of reality on an 18-link H200**, so a
saturated NCCL ring rendered as a nearly idle fabric. Confirmed three independent ways:

- `nvml.h:1697-1705`, verbatim: *"Link ID needs to be specified in the scopeId field in
  nvmlFieldValue_t. A scopeId of UINT_MAX returns aggregate value summed up across all
  links for the specified counter type in fieldId."*
- pynvml's own source: `try: (values[i].fieldId, values[i].scopeId) = fieldId / except
  TypeError: values[i].fieldId = fieldId`.
- the collector demonstrably passed bare ints.

Now requests the aggregate scope explicitly, falling back to summing per-link scopes on a
driver that rejects it. **The test fake mirrored the old bare-int call shape**, which is
why no test could see this; it now honours `scopeId` like real NVML.

**2. The cgroup-v2 OOM guard measured the job against its own REQUEST. (MED, 3 votes)**
v2 spells "no enforced limit" as the string `"max"`; v1 uses a huge sentinel that the
`> 10**16` branch converts to node RAM. Only the sentinel was recognised, so on a v2
`ConstrainRAMSpace=no` node `cgroup_limit` silently became the allocation and a job merely
over its *request* tripped a false "near limit, raise `--mem`" critical — the exact
regression P3 removed, and **the opposite verdict from v1 for identical machine state**.

**3. CPU rates were computed across incomparable counters. (MED, 3 votes)**
`_read_cpu_ns` answers from v2 `cpu.stat`, v1 `cpuacct.usage`, or the `/proc` accumulator,
and the baseline stored whatever arrived. Differencing a 1002 s cgroup counter against a
20 s `/proc` accumulator produced **`effective_cores = 982` on a 4-core job**, which then
latched into `peak_effective_cores` for the rest of the session. The same line also stored
`None` on an unreadable frame, discarding a good baseline and costing **two** consecutive
frames of 0% on a busy job. Now tracks which counter produced the baseline, re-seeds on a
source change, and keeps the baseline across a blind frame.

**4.** `links_per_gpu` took `max(enabled, LINK_COUNT)` — but `LINK_COUNT` is documented as
links **present**, so a degraded fabric advertised spec bandwidth ("18 links · 900 GB/s")
while the topology grid beside it said `NV17`.
**5.** The per-device throughput lists appended a hard `0.0` for a device that had only
just seeded its baseline while its siblings published real rates, halving a summed fabric
total and drawing that GPU as idle.
**6.** `_build_topology` took a cached device index of `-1` at face value where
`_collect_gpus` falls back to the handle position, so one GPU's counters were read twice
under two names and another's never.
Plus: removed a dead `_proc_cpu_ns` (zero callers, and precisely the non-monotonic
live-PID sum the accumulator replaced — an inviting footgun) and a truncated comment.

### `58e74eb` — pending/tui: stop advising a requeue onto capacity that cannot run

**7. `sinfo -O` output is fixed-WIDTH, not delimited. (MED)** It truncates each value to
the field width and pads only *up to* it, so a value within a character of the width
merges with its neighbour. **Verified against live sinfo:** narrow widths print
`testmixgpu:4 gpu:4` as one token. The free-GPU parser split on whitespace runs, so a long
GRES string yielded 3 fields instead of 4, `GresUsed` read as empty, and **a node with
every GPU allocated was counted as entirely free**. Each field now carries an explicit
`|` suffix, and a GPU node whose `GresUsed` is missing is skipped rather than guessed at.

**8. `gpu_detail` conflated "query failed" with "no schedulable GPU node". (MED)** It was
set only for partitions producing ≥1 idle-or-mixed GPU node line, so a partition whose GPU
nodes were all allocated looked like *missing data* — and a GPU job then fell back to the
partition's idle **GPU-less** node count, printing "FITS NOW" and a copy-pasteable
`scontrol update` for a partition with **zero** free GPUs.

**9. The WHERE tip's condition was a tautology. (MED)** It tested `fits`, which is
deliberately forced `False` for the current partition, making `not any(...)` always true:
the "no partition currently has enough free capacity" claim was emitted unconditionally
and the `else` was unreachable. A job pending on `Reason=Priority` with 8 free nodes and
240 idle cores in its own partition was told there was no capacity — contradicting the
columns directly above it. *(Mutation-proven: reverting the fix reproduces the message.)*

**10.** The job selector could hide every column but NAME — its NAME cell was the only
name-render site skipping `_elide_job_name`, and `_column_widths` sizes the column to the
longest name, so one ordinary 68-char sweep name pushed the header, rule and every row
past the terminal edge.
**11.** `_poll_jobs`' widget lookups are now guarded as a group: that coroutine resumes
from awaits when the screen may already be dismissed, and an unguarded `query_one` raises
`NoMatches`, which `run_worker`'s default `exit_on_error` escalates to `WorkerFailed` —
**taking the whole app down on a keypress**.
**12.** JOB and PENDING card border titles hard-coded a Unicode `·` where the third
sibling gates it, breaking `--ascii` purity. *(Mutation-proven.)*

### `bc62f66` — cli: never launch an invisible TUI; exit cleanly on a pipe or Ctrl-C

**13. `sw <job>` with stdout redirected blocked forever. (MED)** No terminal check, so a
batch script's `sw $SLURM_JOB_ID >> mon.log` launched the dashboard with nobody to draw
for and no keypress able to quit it. **Measured: rc=124 at timeout, 0 bytes on stdout,
323693 bytes of ANSI in stderr per 20 s.** Now emits one snapshot plus guidance — the same
degradation the pending, hop and ssh paths already make (they share the "a TUI needs a
terminal" test; only this one lacked it). Verified: rc=0, a 2190-byte CSV snapshot.

**14. The `--once` broken-pipe guard could never fire. (LOW-MED)** Piped stdout is
block-buffered, so a small payload never touched the pipe inside the `try` and EPIPE
surfaced at interpreter-shutdown flush — outside the handler. `--once --json` into an
early-closing reader exited **120** with `Exception ignored ... BrokenPipeError`. Verified
fixed: 120 → 0, silent.

**15. An empty job id silently monitored a different job. (LOW-MED)** `""` is not `None`,
so it skipped auto-discovery and ran `scontrol show job -d ""` — which means *every* job.
slurmwatch resolved whatever came back (the caller's own `$SLURM_JOB_ID` allocation) and
emitted real telemetry tagged with the empty id, leaving a `--log` CSV with a blank
`job_id` primary key on every row. `sw "$JOBID" --once` with `JOBID` unset was enough.

**16.** Ctrl-C on the machine paths printed a raw traceback; the hop/ssh paths already
exited 130 cleanly.

### `e5db694` — output: job name in text summaries, complete CSV schema

**17. The job name reached every TUI surface, `--json` and CSV, but none of the three
plain-text summaries.** That is exactly the report a user redirects to a file from a login
node. `git log -S job_name -- src/slurmwatch/cli.py` returns **nothing** — the two earlier
"parsed but never shown" fixes never touched `cli.py`. This is the blind-spot class a
number-accuracy audit structurally cannot find.
**18.** Three fields were `--json`-only while **CSV is the default for `--once`**:
`gpu_monitoring_available` (on a ROCm/oneAPI node the CSV showed `gpu_count=0` with no
flag, so a script reads a measured zero and advises dropping the GPU request),
`gpu_<N>_throttle_reasons`, and `cpu_usage_ns`.
**19.** CSV formula injection: quoting does not stop a spreadsheet **evaluating** a cell
beginning `= + - @`, and a job name is arbitrary user text.
**20.** `from_dict` crashed on its own output — `to_json` maps a non-finite to `null`, and
feeding that back raised `TypeError`, which `parse_snapshot_line` swallows as
"unparseable", so the node switcher silently showed that node as producing **no data at
all**, with no diagnostic.
**21.** A pre-existing **flake**: `test_pending_app_mounts_and_refreshes` polled only
~0.9 s for an executor-backed resolve then asserted against a still-empty view — failed
about 1 run in 5 on a loaded machine (runtimes 0.35–2.34 s observed).

### `6fd0112` — gpu: stop presenting an unreadable reading as a measured zero

**22. VRAM / power / temperature had no availability flag. (MED)** All three initialise to
0 and each read could fail silently — but 0 is a plausible reading for all three, so a
failed read was published as fact. Utilization already carried `utilization_available` for
exactly this reason. Worst consequence: the activity heuristic vetoes on
`memory_used_bytes > 0`, so an unreadable VRAM query scored a GPU at **99% utilization as
IDLE** — amber "idle", `0 active` in the head glance, `gpu_active_count=0`, and
`memory_utilization_percent: 0.0`, i.e. *"0% HBM, raise your batch size"* on a full card.
Reachable and persistent: nvidia-ml-py ≥ 11.510 raises `FunctionNotFound` from
`nvmlDeviceGetMemoryInfo_v2` against a pre-510 driver. On the display side a MIG slice
rendered `0 W · 0°C (32°F)` — an unpowered, below-freezing card the collector had
simultaneously scored active — with the °F conversion dressing the fabricated zero up as
a second, independently-derived measurement.

### Test-coverage gaps closed (mutation sweep, A4)

34 mutations, **30 caught, 4 NOT caught**. Every caught mutation failed exactly one test —
a healthy one-regression-test-per-fix signature. The four gaps, all now covered:

| Behaviour | Why it mattered |
|---|---|
| excluding `os.getpid()` from the `/proc` CPU sum | The `/proc` sum **is** the production CPU path on a cpuacct-less cluster, and after an `srun` hop sw runs *inside* the job's cgroup — its own CPU would bill to the job and latch into `peak_effective_cores` |
| the `sinfo` state-flag filter (`* $ % @ !`) | `base = re.sub(r"[^a-z]","",state)` strips flags *before* the idle/mix test, so the flag check is the only thing excluding an unreachable `idle*` node from the free-GPU pool |
| `_norm_bus`'s PCI-domain normalisation | Its loss silently downgrades genuine NVLink hardware to `fabric="pcie"` with no generation and no bandwidth |
| `allow_nan=False` on `to_json` | `_json_safe` only walks float/dict/list, so a non-finite in a tuple or a numpy scalar still emits a bare `NaN` and makes `jq` reject the whole `--log` file |

---

## NOT fixed — verified findings, deliberately deferred

These are real and reproduced. They are deferred because each needs a careful,
independently-tested change rather than being folded into a sweep. Worst first.

| # | Sev | Finding | Why deferred |
|---|---|---|---|
| D1 | MED | **`/proc` accumulator drops CPU time of processes it never sampled.** `_parse_stat_cpu_ticks` uses `utime+stime` only (no `cutime/cstime`), and ticks are retained only for PIDs seen in ≥1 poll, so a child born AND dead inside one poll interval contributes **zero**. Probe: true 12.20 s vs reported 0.20 s — a **98.4% under-report** ("you're using 0.04 of 4 cores, lower `-c`" on a job genuinely using 2.4). This is the production CPU path on Midway3. Second shape: a PID reused by a *busier* process takes the `cur >= prev` branch and loses 400 of 1340 ticks. | The fix is `cutime/cstime` reconciliation, which double-counts if done naively. Needs its own change with dedicated tests. |
| D2 | MED | **The per-sample executor future is awaited with no timeout.** One wedged NVML/procfs call freezes telemetry permanently *and silently* (every gauge and the elapsed clock latch — indistinguishable from an idle job), then `asyncio.run`'s `shutdown_default_executor()` joins the wedged thread and the process hangs invisibly at exit. `_bounded_exit` exists for exactly this and is used on `--once`/headless but not the TUI path. | Needs a staleness indicator designed alongside it (D3), not a bare timeout. |
| D3 | MED | **No staleness cue when a stream dies after a successful attach.** Dashboard widgets re-render only when a frame arrives and the stuck-switch watchdog runs only while switching, so the "Ns old" tag *freezes at its last value* and 5-minute-old gauges read as live. The drill-in recomputes age every 0.5 s → the same fact, two different answers on two screens. | Same design as D2; do them together. |
| D4 | MED | **An unreachable remote node respawns two `srun`s forever.** `min(2.0 ** min(fails-1, 3), 8.0)` caps the backoff at 8 s with no escalation and no give-up, and `open_stream` re-runs the GPU probe every attempt → **~10k–21k `srun` step-create RPCs/day** against slurmctld, unattended. | Needs a give-up policy and a per-node probe cache. |
| D6 | MED | **"peak" is since-sw-attach and never labelled as such.** `peak working set` / `peak N cores` / `mem_peak_working_set_bytes` / `cpu_peak_effective_cores` are running maxima since attach; only `model.py` concedes it. In `--once` (one sample) they are arithmetically identical to the instantaneous reading yet still called "peak" — one field away from `mem_peak_bytes`, which *is* a true kernel lifetime peak. The documented right-sizing workflow therefore under-sizes `--mem`. | A labelling/semantics decision (and possibly a `samples` field) for the author. |
| ~~D23~~ | LOW-MED | **FIXED** (working tree) — see `FIXED and shipped`. Re-measured before touching anything, because nine items in this family have now been found already fixed while the row still called them open: this one was **live**, and the row's numbers hold exactly. A real 1.00 s window in the spinner state gave **8 ticks, 8 `render()` calls, 8 `sacctmgr` spawns**; a `sacctmgr` taking 200 ms to fail held the event loop for **1.00 s of a 1.14 s window** and cut the spinner to 5 ticks. Fixed by the second of the two shapes this row named — the resolve moved into the 10 s poll's executor, `pending.py`'s no-cache-on-failure policy untouched — because a negative cache bounds the rate of a stall in `render()` but not its 15 s duration. | |
| D8 | LOW-MED | **MIG device indices are unique only within a parent**, but `_nvml_handle_info` / `_handle_for_device` key on the NVML index, so two slices of one GPU both answer index 0: the second's uuid/name overwrite the first's (`gpu_0_uuid == gpu_1_uuid`) and both map to one handle. The shared parent bus id also collapses `pos_by_bus`, giving two slices of one card a bogus 2-device fabric grid. | Key the caches on handle position; no MIG hardware here to validate against. |
| D9 | LOW-MED | **`cores_allocated` is Slurm's allocated *logical CPUs*** (`NumCPUs` / `CPU_IDs` enumerate threads) but every label says "cores". On `ThreadsPerCore=2` a job with 2 physical cores reads "of 4 cores" and the user chases phantom SMT headroom. Invisible on Midway3 (HT off). | Rename to "CPUs" (matches Slurm and `--cpus-per-task`) or read `ThreadsPerCore` — a wording decision. |
| D10 | LOW | **Free-GPU counting is type-blind.** `_sum_gres_gpus` discards the captured type and `fit_blocker`'s type gate is partition-level only, so on this cluster a `--gres=gpu:a30:1` job counts free GPUs on ~88 *untyped* nodes → "FITS NOW" while every a30 is busy. | Wants a per-type free map. |
| D11 | LOW | **Interconnect topology is latched on the first GPU frame even when the probe FAILS** (`_interconnect_probed = True` is set *before* the probe and never reset), so a transient first-frame NVML error pins `fabric="pcie"` — or `None`, hiding the section — for a multi-day run. | Latch only on success, with a bounded retry budget. |
| D12 | LOW | **A device NVML can't identify at attach time shifts every later `cuda_ordinal`**, so a 3-GPU job whose middle card has fallen off the bus labels its `cuda:2` as `CUDA 1` everywhere. | Record the requested-list position at attach. |
| D13 | LOW | **No memory fallback in the v1 branch.** The `_proc_rss_bytes` fallback exists only in the v2 branch and there is no `else`, so `v1_cpu` set + `v1_mem` absent → MEM reports a confident `0 B / 0.0%`, and because discovery *succeeded* it never degrades to `sstat`. A user right-sizing from that would cut `--mem` and OOM. | Needs an `else` arm plus a "leave unset so the caller degrades" signal. |
| ~~D14~~ | LOW | **FIXED** (working tree) -- see `FIXED and shipped`. Re-measured before touching anything, because nine items in this family have now been found already fixed while the row still called them open: this one was **already closed**, and the row is what was stale. `_job_ended` is read at `tui.py:2771` inside `ResourceDetailScreen._refresh`, the title now reads `job ended` on-node and `42s old - job ended` off-node, and `git show HEAD:src/slurmwatch/tui.py` has **zero** occurrences of the phrase against four in the tree. The comment at the fix site carries the account, which is the fastest tell -- as it was for D15 and D22. No code change this round. | |
| ~~D15~~ | LOW | **STALE — fixed AND tested at HEAD; the row was never updated.** Re-measured end-to-end before touching anything, because seven items in this family were already fixed while the row still called them open: this one was **already closed**. See `Verified CLEAN` for the numbers and the teeth run that proves the guard load-bearing. No code change. | |
| ~~D16~~ | LOW | **FIXED** (working tree) — see `FIXED and shipped`. Re-measured before touching anything, because six items in this family were already fixed while the row still called them open: this one was **live**. `grep -rn '\.qos\b' src/slurmwatch/` returned 5 reads, every one of them `JobContext.qos` (the running and foreign cards), `cli.py` returned zero `qos` lines at all, and `git show HEAD:src/slurmwatch/pending.py` confirmed the same write-only field at HEAD. Both pending renderers now print the parsed QOS beside the reason code. | |
| ~~D17~~ | LOW | **FIXED** (working tree) — see `FIXED and shipped`. Re-measured first, because five items in this family were already fixed: this one was **live**. `grep -rn '\.total_nodes\|\.cpus_total' src/` returned only the two `+=` that write them, and two rendered rows with `cpus_idle=240` over `cpus_total` 256 vs 3200 were byte-identical after the partition name on **both** surfaces. Both capacity columns now read `free/total` through one shared helper. | |
| ~~D18~~ | LOW | **FIXED** (working tree) — see `FIXED and shipped`. Re-measured first, because six items in this family were already fixed: this one was **live**, and the "self-heals on the node" half is **corrected**. Off-node, one `scontrol` record with `gres/shard=2` gave `gpu_count_requested=0` (control: `gres/gpu=2` → 2) and the dashboard rendered `● GPU     none requested` with the `g` drill-in reading `no GPUs requested by this job`; `git show HEAD:` has no `shard`/`mps` in `slurm.py`, `model.py`, `tui.py` or `cli.py`. The node heals it ONLY through a job process's `CUDA_VISIBLE_DEVICES` — 1 device with it exported, still **0** with the environ unreadable — because `_GRES_IDX_RE` needs a literal `gpu` and never matches `GRES=shard:2(IDX:0)`. Both dashboard sites and the plain report now name the request (`shard:2 requested — a fraction of a device`) instead of asserting a zero nothing measured. | |
| ~~D19~~ | LOW | **FIXED** (working tree) — see `FIXED and shipped`. Measured: of 11 widget-class selectors in the app's CSS exactly 2 (`Footer`, `FooterKey`) were unbacked by any import or local class; 3 of the 5 widgets the comment credited (`Sparkline`, `DataTable`, `Rule`) are imported nowhere. | |
| D20 | LOW | **NOT REACHABLE while the cap holds; the condition is now pinned.** The asymmetry is real — the constants table takes `ThrottleReason*` or `EventReason*`, the getter takes one spelling — but `pyproject.toml` declares `pynvml>=11.5,<12` and the `EventReason*` names arrived in **12**. Measured on the installed 11.5.3: `nvmlDeviceGetCurrentClocksThrottleReasons` and `nvmlClocksThrottleReasonSwPowerCap` present, both `*EventReason*` spellings absent. `tests/test_nvml_spelling_coverage.py` asserts `cap <= 12 or the getter accepts EventReasons` — raising the cap without fixing the getter fails, raising it WITH the fix passes (verified both ways). `collector.py` unchanged. | |
| ~~D21~~ | LOW | **WITHDRAWN as written, and replaced by a real gap.** It does read the real `$HOME`, but it does not break: `/`, empty, unset, a trailing slash and a relative value all pass (`expanduser` falls back to the passwd entry, and at `HOME=/` the test's own `home + "/work/run.sh"` yields the doubled slash the guard matches). The genuine hole was the NEGATIVE — nothing checked that the collapse stops at a component boundary. Pinned now; see `FIXED and shipped`. | |
| ~~D22~~ | LOW | **STALE — fixed AND tested; the row was never updated.** `_escape_markup` now calls `printable_text(text)` first, with the reason inline: "Rich passes ESC straight through to the terminal (measured), so a job name carrying `\x1b[2J` would clear the dashboard." Measured 2026-09-05: **zero raw C0 bytes survive** either `_escape_markup` or `printable_text`. Covered by SW-82 in `test_cli.py`, which drives `HOSTILE = "train\x1b[2J\x1b[31mRED\rboom\x08x"` through `_csv_text`, `_name_suffix` **and** `_escape_markup`, plus a control that accented and CJK names survive intact. Swept the adjacent question: every job-name render site is guarded — `tui.py:2440` and `tui.py:5924` both go through `_escape_markup(_elide_job_name(...))`. | |

---

## Verified CLEAN — useful negatives

Recorded so future passes don't re-audit them:

- **The remote text summary's `peak` label is correct in BOTH branches** — checked because
  the no-limit line hardcodes `peak` while the dashboard picks `"peak" if snap.remote else
  "used"`. `_print_remote_summary` is the off-node path and its only caller is the remote
  one, so the figure genuinely *is* a lifetime MaxRSS there; there is no on-node text
  summary to disagree with it. Not a drift.
- **`pending.py`'s reason table folds completely** — all 21 entries through
  `explain_reason(..., ascii_mode=True)` leave zero non-ASCII, and the only glyph the table
  actually contains (U+2014) is one `_asciify` handles. `_REASON_EXPLANATIONS` is private
  with a single reader, so there is no second door around the fold.

- **CSV width agreement has no mismatch** — 108+ combinations of 0–17 GPUs × `max_gpus` ∈
  {n, 0, 1, 8, 17, None} all agree; cross-append drift logic and `_warn_csv_schema_drift`
  both work.
- **`--json` drops no field** — all snapshot fields restored, `asdict(before) ==
  asdict(after)` lossless including quotes/newlines in `job_name`.
- **argparse and exit codes are disciplined** — unknown flag → 2, `--interval 0/-5/nan` →
  2, `--once --log` → 1, no-Slurm → 1, finished job → 1, all with **0 bytes on stdout**.
- **Growth at defaults is constant-memory** — measured, not projected: ~44 MB RSS, two
  800-sample windows differing by 56 kB (the rest is Textual's internal LRU). D5 is the
  only unbounded-in-config dimension.
- **No pipe-full deadlock possible** (stderr is `DEVNULL`, only stdout is piped); orphan
  reaping is complete; the exit-time `Event loop is closed` fix is complete; the
  late-reply node-switch race is genuinely closed; **no `except` swallows
  `CancelledError`**; every overlappable worker is `exclusive=True`.
- **Dependency floors are honest** — pynvml 11.5.3 has every symbol used; nothing newer
  than Python 3.10 (newest construct is `zip(strict=True)`).
- **Portability**: `--mem=0` → MemTotal (correct, guarded); cgroup v1 *and* v2 including
  Slurm 25.05+/26.x SLUID names with three fallbacks; no hard-coded hostname pattern;
  locale pinned (`LC_ALL=C`, `SLURM_TIME_FORMAT=standard`); `SC_CLK_TCK`/`SC_PAGE_SIZE`
  read, not assumed; every Slurm flag used predates 20.11.
- **The selector's return-cursor DOES follow the live-refreshed list** (D15, withdrawn).
  The symptom is real and the mechanism is exactly as described — but the adoption is
  already at HEAD, so there was nothing to fix. Measured end-to-end rather than read:
  drove the picker with a list that changes between the first render and the return —
  run-loop snapshot `['111','12345']`, live refresh → `['111','12345','999']`, opened
  index 2 → job **999**, quit the dashboard, and the cursor came back to **index 2 → job
  999**. The single line that does it is `tui.py:6457` (`jobs, sampled_at =
  screen.sample`), and `git diff HEAD` touches no line of `_run_job_selector`, so it is
  HEAD's, not a working-tree edit. **Teeth, so this negative is not just a passing test:**
  replacing that one line (1 occurrence, asserted before and after) with `_dropped,
  sampled_at = ...` reproduces the row *verbatim* — the user opens 999 and the cursor
  parks on **index 0 → job 111**, because the lookup falls back to the pre-picker
  snapshot — and reddens exactly **1** of 482, `test_tui.py::TestJobSelectorFlow::
  test_cursor_restored_for_a_job_that_only_exists_via_live_refresh` (`assert 0 == 2`).
  481 pass under the neuter, including the sibling non-refresh cursor case and
  `test_returning_to_the_picker_carries_the_sample_time_with_the_list` — the neuter is
  scoped to the *cursor* half of `sample`, leaving the timestamp half intact.
- **`_job_anchor` being unescaped is safe** — Textual builds the header with
  `Content(title)`, which is literal.
- **No skipped, xfailed or assertion-free tests**; `conftest.py` is hermetic.
- `/proc/<pid>/stat` is thread-group aggregated, so multi-threaded jobs are counted
  correctly; `dt` can never be negative; `_parse_mem_to_bytes` is correct on every shape;
  peak monotonicity holds on both paths.

---

## Status

Every finding in the FIXED section is remediated and covered by a test, verified on this
hardware (Midway3: H100 PCIe and H200 SXM nodes, cgroup v1, no per-job cpuacct). The
NOT-FIXED table is an honest backlog of verified defects, not a list of regressions — and
the new observation surfaces this pass opened (mutation testing of guard coverage;
omission/field-diff analysis; docs-vs-behaviour) are why it found things five previous
passes did not.
