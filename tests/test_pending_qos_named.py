"""D16: the pending job's own display never said WHICH QOS was throttling it.

``PendingJob.qos`` is parsed off the same ``scontrol show job`` record as every other
field of that dataclass and was then read by nothing in ``src/`` — while the
running-job card (:class:`~slurmwatch.tui.JobDetailsPanel`) and the foreign-job card
(:meth:`~slurmwatch.tui.ForeignJobView._provenance`) have both carried a ``qos`` chip
all along. So a job sitting on ``QOSMaxJobsPerUserLimit`` was told "A QOS limit is
capping your usage" by the one surface that knew the QOS name and refused to print it.

The pending report is rendered TWICE — the dashboard (:class:`PendingView`) and the
plain-text report (``cli._print_pending_summary``) — and "fixed on one side only" is
this repo's recurring defect, so the two surfaces are asserted in separate classes:
neutering one renderer must redden only that renderer's class.
"""

from __future__ import annotations

import io

import pytest
from rich.text import Text

import slurmwatch.cli as cli
from slurmwatch import pending
from slurmwatch.config import SlurmwatchConfig
from slurmwatch.model import JobContext
from slurmwatch.pending import PendingJob
from slurmwatch.tui import ForeignJobView, JobDetailsPanel, PendingView

# A QOS name that cannot be confused with anything else this report prints (a reason
# code, a partition name, an explanation sentence).
_QOS = "gpu-throttle"


def _job(**overrides: object) -> PendingJob:
    """The demo pending job, with fields overridden per test."""
    job = pending._mock_pending_job("777")
    for key, value in overrides.items():
        setattr(job, key, value)
    return job


def _view(job: PendingJob) -> PendingView:
    """A ``PendingView`` rendered unmounted, as every other PendingView test does."""
    view = PendingView()
    view.job = job
    view.config = SlurmwatchConfig()
    return view


def _why(job: PendingJob, ascii_mode: bool = False) -> str:
    """The dashboard's "Why It's Waiting" band, markup stripped."""
    return Text.from_markup(_view(job)._why(job, ascii_mode)).plain


def _report(monkeypatch: pytest.MonkeyPatch, job: PendingJob, ascii_mode: bool = False) -> str:
    """The plain-text report with all four Slurm lookups stubbed out."""
    monkeypatch.setattr(cli, "resolve_priority_rank", lambda *a: None)
    monkeypatch.setattr(cli, "resolve_queue_counts", lambda *a: None)
    monkeypatch.setattr(cli, "resolve_cluster_partitions", lambda *a: [])
    monkeypatch.setattr(cli, "resolve_user_associations", lambda *a: None)
    buf = io.StringIO()
    cli._print_pending_summary(job, stream=buf, ascii_mode=ascii_mode)
    return buf.getvalue()


class TestTheDashboardNamesTheThrottlingQos:
    """Surface 1 of 2: ``tui.PendingView``."""

    def test_the_qos_sits_beside_the_reason_code_that_names_one(self) -> None:
        plain = _why(_job(qos=_QOS, reason="QOSMaxJobsPerUserLimit"))
        # The reason code and the explanation both say "QOS" and neither says which.
        assert "QOSMaxJobsPerUserLimit" in plain
        assert f"qos {_QOS}" in plain, plain

    def test_the_qos_survives_ascii_mode(self) -> None:
        # --ascii folds the separators, not the field: a non-UTF-8 terminal must not
        # be the one place the QOS goes missing.
        plain = _why(_job(qos=_QOS, reason="QOSMaxJobsPerUserLimit"), ascii_mode=True)
        assert f"qos {_QOS}" in plain, plain

    def test_the_qos_is_shown_for_a_non_qos_reason_too(self) -> None:
        # It is identity, not an annotation on the reason — the running and foreign
        # cards show it unconditionally, so a Resources-blocked job gets it as well.
        plain = _why(_job(qos=_QOS, reason="Resources"))
        assert f"qos {_QOS}" in plain, plain

    def test_a_bracket_in_the_qos_name_cannot_break_the_markup(self) -> None:
        # Everything user/site-supplied that reaches Textual's parser is escaped;
        # an unescaped '[' raises MarkupError instead of rendering.
        plain = _why(_job(qos="odd[qos", reason="Resources"))
        assert "odd[qos" in plain, plain

    def test_no_dangling_label_when_slurm_reports_no_qos(self) -> None:
        # Passes with the fix in OR out: an absent value must print no chip and no
        # orphaned separator, exactly as before.
        plain = _why(_job(qos="", reason="Resources"))
        assert "qos" not in plain.lower(), plain
        assert "PENDING" in plain


class TestThePlainReportNamesTheThrottlingQos:
    """Surface 2 of 2: ``cli._print_pending_summary``."""

    def test_the_header_line_carries_the_qos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        head = _report(monkeypatch, _job(qos=_QOS, reason="QOSMaxJobsPerUserLimit")).splitlines()[0]
        assert f"qos {_QOS}" in head, head
        # Still the identity line it always was.
        assert "Job 777" in head and "PENDING" in head

    def test_the_header_line_carries_the_qos_in_ascii_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = _report(monkeypatch, _job(qos=_QOS, reason="QOSMaxJobsPerUserLimit"), True)
        assert f"qos {_QOS}" in text.splitlines()[0], text

    def test_the_qos_is_shown_for_a_non_qos_reason_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = _report(monkeypatch, _job(qos=_QOS, reason="Resources"))
        assert f"qos {_QOS}" in text.splitlines()[0], text

    def test_no_dangling_field_when_slurm_reports_no_qos(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Passes with the fix in OR out.
        head = _report(monkeypatch, _job(qos="", reason="Resources")).splitlines()[0]
        assert "qos" not in head.lower(), head
        assert "Job 777" in head and "PENDING" in head


class TestNeitherSurfaceIsLeftBehind:
    """The cross-surface assertion — it reddens under EITHER neuter, by design.

    The two classes above attribute a break to one renderer; this one is the detector
    for the defect the repo keeps hitting, where a fact reaches the dashboard and
    never the report a user redirects to a file.
    """

    def test_both_renderers_name_the_same_qos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        job = _job(qos=_QOS, reason="QOSMaxJobsPerUserLimit")
        needle = f"qos {_QOS}"
        dashboard = needle in _why(job)
        report = needle in _report(monkeypatch, job)
        assert dashboard and report, (
            f"the QOS reached the dashboard={dashboard} and the plain report={report} "
            "— one surface names the throttling QOS and the other still does not"
        )


class TestControlsUnchangedByTheFix:
    """None of these read the code the fix added; all pass with the fix in or out."""

    def test_the_dashboard_still_shows_the_reason_its_explanation_and_the_request(
        self,
    ) -> None:
        plain = _why(_job(qos=_QOS, reason="Resources"))
        assert "Why It's Waiting" in plain
        assert "Resources" in plain
        assert "free nodes" in plain  # explain_reason("Resources")
        assert "requested (total)" in plain
        assert "16 CPU" in plain and "2x a100" in plain

    def test_the_plain_report_still_reads_in_its_documented_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        text = _report(monkeypatch, _job(qos=_QOS, reason="Resources"))
        order = [text.index(k) for k in ("Job 777", "Why", "When", "Needs")]
        assert order == sorted(order), text
        # The name suffix is the last thing on the identity line, as before.
        assert text.splitlines()[0].endswith("name `train`"), text

    def test_the_running_card_still_names_the_qos(self) -> None:
        # The half of the asymmetry that always worked, and must keep working.
        panel = JobDetailsPanel()
        panel.job_ctx = _ctx()
        panel.config = SlurmwatchConfig()
        plain = Text.from_markup(panel.render()).plain
        assert "qos normal" in plain, plain

    def test_the_foreign_card_still_names_the_qos(self) -> None:
        plain = Text.from_markup(ForeignJobView()._provenance(_ctx(), False)).plain
        assert "qos normal" in plain, plain


def _ctx() -> JobContext:
    """A running job's context, shaped like ``test_tui._provenance_ctx``."""
    return JobContext(
        job_id="12345",
        username="ada",
        partition="gpu",
        nodelist="cn001",
        hostname="cn001",
        cpus_allocated=16,
        mem_limit_bytes=64 * 1024**3,
        gpu_count_requested=2,
        gpu_indices=[0, 1],
        job_name="train-llama-8b",
        account="rcc-staff",
        qos="normal",
        job_state="RUNNING",
        command="/home/ada/proj/train.py",
        work_dir="/home/ada/proj/runs",
        submit_time=1000.0,
        job_start_time=1180.0,
    )
