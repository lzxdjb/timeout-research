"""Progress remains readable in terminals and logs without contaminating JSON stdout."""

import io

import pytest

from scripts.audio_data import progress


@pytest.fixture
def clock(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: now[0])
    return now


def test_redirected_on_progress_is_throttled_and_unknown_total_has_no_percentage(monkeypatch, clock, capsys):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    with progress.Progress("Scan metadata", mode="on", interval=2) as indicator:
        assert "0 records" in stream.getvalue()
        clock[0] = 1
        indicator.update(3)
        assert len(stream.getvalue().splitlines()) == 1
        clock[0] = 2.1
        indicator.update(2)
        assert "5 records" in stream.getvalue()
        assert "2.1s elapsed" in stream.getvalue()
    assert "%" not in stream.getvalue()
    assert "\r" not in stream.getvalue()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("mode", ["auto", "off"])
def test_redirected_auto_and_off_are_silent(monkeypatch, mode):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    with progress.Progress("Rows", mode=mode, total=10, interval=0) as indicator:
        indicator.update(10)
    assert stream.getvalue() == ""


def test_terminal_auto_bar_reports_resume_offset_without_inflating_rate(monkeypatch, clock):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    stream = Terminal()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    with progress.Progress("Resume scan", mode="auto", total=10, initial=3, interval=0) as indicator:
        assert "30.0%" in stream.getvalue()
        clock[0] = 2
        indicator.update(2)
    result = stream.getvalue()
    assert "50.0%" in result
    assert "1 records/s" in result
    assert "[" in result
    assert "\r" in result
    assert result.endswith("\n")


def test_total_can_be_discovered_and_close_is_idempotent(monkeypatch, clock):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    indicator = progress.Progress("Extract", mode="on", interval=0)
    clock[0] = 1
    indicator.update(5)
    indicator.set_total(10)
    assert "5/10 50.0%" in stream.getvalue()
    indicator.close()
    result = stream.getvalue()
    indicator.close()
    assert stream.getvalue() == result


def test_byte_progress_is_readable_and_completion_renders_inside_short_tasks(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    with progress.Progress("Checksum", mode="on", total=2048, unit="bytes", interval=1000) as indicator:
        indicator.update(2048)
    assert "2.00 KiB/2.00 KiB 100.0%" in stream.getvalue()


def test_progress_finally_closes_when_work_raises(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(progress.sys, "stderr", stream)
    with pytest.raises(ValueError, match="source error"):
        with progress.Progress("Check", mode="on", total=10, interval=1000) as indicator:
            indicator.update(2)
            raise ValueError("source error")
    assert "2/10 20.0%" in stream.getvalue()
    assert indicator.closed
