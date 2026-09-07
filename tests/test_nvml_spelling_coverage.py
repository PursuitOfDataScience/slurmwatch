"""The throttle CONSTANTS accept two pynvml generations; the GETTER accepts one.

`collector._check_gpu_throttling` builds its bitmask from a table that deliberately
takes either spelling, and says why:

    # (label, constant spellings) — names differ across pynvml releases
    # (ThrottleReason* vs the newer EventReason*). Same 5 bits as before.

The call that produces the bitmask does not follow that rule. It is
`pynvml.nvmlDeviceGetCurrentClocksThrottleReasons` only, wrapped in
`except (pynvml.NVMLError, AttributeError): pass`, so on a pynvml that renamed it the
function returns `(False, [])` — indistinguishable from a GPU that is not throttling.
That is `issues.md` D20.

**Measured: not reachable today, and the cap is the only reason.** `pyproject.toml`
declares `pynvml>=11.5,<12`; the `EventReason*` names arrived in 12. On the installed
11.5.3, `nvmlDeviceGetCurrentClocksThrottleReasons` and
`nvmlClocksThrottleReasonSwPowerCap` are present and both `*EventReason*` spellings
are absent — so within the declared range the single getter spelling always resolves.
The forward-looking half of the constants table is dead code that costs nothing.

So this file does not change the getter. It pins the *condition* that would turn the
asymmetry into D20: **if the cap is ever raised to admit pynvml 12, the getter must
gain the second spelling in the same edit.** Nothing else would catch that — the
`AttributeError` is suppressed, and there is no GPU in CI to notice a permanently
false `throttling`.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
COLLECTOR = ROOT / "src" / "slurmwatch" / "collector.py"

#: The release that renamed these, and therefore the version the getter's single
#: spelling stops being sufficient at.
RENAMED_IN_MAJOR = 12


def pynvml_cap_major() -> int:
    """The exclusive upper bound on pynvml, as a major number."""
    found = re.search(r'"pynvml>=[\d.]+,<(\d+)"', PYPROJECT.read_text())
    assert found, "pynvml should still be declared with an upper cap"
    return int(found.group(1))


def getter_generations() -> set[str]:
    """Which spellings of the throttle-reasons GETTER the collector calls."""
    return set(re.findall(r"nvmlDeviceGetCurrentClocks(\w+?)Reasons", COLLECTOR.read_text()))


def constant_generations() -> set[str]:
    """Which spellings of the throttle-reason CONSTANTS the collector accepts."""
    return set(re.findall(r"nvmlClocks(\w+?)Reason[A-Z]", COLLECTOR.read_text()))


class TestTheGetterCoversTheDeclaredPynvmlRange:
    def test_one_getter_spelling_is_enough_only_while_the_cap_excludes_the_rename(self) -> None:
        cap, getters = pynvml_cap_major(), getter_generations()
        assert cap <= RENAMED_IN_MAJOR or "Event" in getters, (
            f"pyproject admits pynvml {cap - 1}.x, which renamed the getter, but the "
            f"collector still calls only {sorted(getters)}Reasons. The AttributeError is "
            "suppressed, so throttling would read False forever instead of unknown "
            "(issues.md D20)."
        )

    def test_the_constants_already_accept_both_generations(self) -> None:
        # The rule the getter does not follow, asserted so the asymmetry is on record
        # rather than rediscovered.
        assert {"Throttle", "Event"} <= constant_generations(), constant_generations()

    def test_both_scans_actually_found_something(self) -> None:
        # A silent regex miss would make the assertion above vacuous.
        assert getter_generations(), "no throttle-reasons getter found in collector.py"
        assert len(constant_generations()) >= 1


class TestControls:
    """None of these compares the cap against the getter, so each holds either way."""

    def test_the_cap_is_still_declared_at_all(self) -> None:
        assert re.search(r'"pynvml>=[\d.]+,<\d+"', PYPROJECT.read_text())

    def test_the_installed_pynvml_is_inside_the_declared_range(self) -> None:
        import pynvml

        version = getattr(pynvml, "__version__", "")
        assert version, "pynvml should report a version"
        assert int(version.split(".")[0]) < pynvml_cap_major(), version

    def test_the_installed_pynvml_has_the_spelling_the_collector_calls(self) -> None:
        # The reason D20 is not reachable today, measured rather than assumed.
        import pynvml

        assert hasattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons")

    def test_the_throttle_check_still_returns_a_pair(self) -> None:
        # Shape only, so this holds whatever the spellings are.
        from slurmwatch.collector import TelemetryCollector

        collector = object.__new__(TelemetryCollector)
        collector._mock = False
        throttling, reasons = TelemetryCollector._check_gpu_throttling(collector, object())
        assert isinstance(throttling, bool) and isinstance(reasons, list)
