"""Run records shared by vistaqa.predict and eval/evaluate.py: time, git state, a log file of the output and a trace."""

from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import logfire
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

REPO = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git_state() -> dict:
    """HEAD commit and the files that differ from it, untracked ones included, outside results/ (the runs)."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout

    status = git("status", "--porcelain", "--", ".", ":(exclude)results")
    return {"commit": git("rev-parse", "HEAD").strip(), "modified": [line[3:] for line in status.splitlines()]}


class _Tee:
    def __init__(self, stream, log) -> None:
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.log.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.stream, name)


def tee_output(log_path: Path) -> None:
    """Append everything printed to stdout and stderr at the Python level to log_path as well.

    Output written before this call (import-time warnings) or through handlers bound to the original streams
    (the transformers logger) reaches only the console.
    """
    log = log_path.open("a", buffering=1)
    sys.stdout, sys.stderr = _Tee(sys.stdout, log), _Tee(sys.stderr, log)
    print(f"=== {now()} {' '.join(sys.argv)}", flush=True)


@contextmanager
def output_scope():
    """Keep sequential invocations in one process from appending to earlier runs' logs."""
    stdout, stderr = sys.stdout, sys.stderr
    try:
        yield
    finally:
        logs = set()
        for stream, original in ((sys.stdout, stdout), (sys.stderr, stderr)):
            while isinstance(stream, _Tee) and stream is not original:
                logs.add(stream.log)
                stream = stream.stream
        sys.stdout, sys.stderr = stdout, stderr
        for log in logs:
            log.close()


class _JsonlSpanExporter(SpanExporter):
    """Appends every finished OpenTelemetry span to a JSON-lines file."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("a", buffering=1)

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            self.file.write(json.dumps(json.loads(span.to_json()), ensure_ascii=False) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self.file.close()


def trace_to(path: Path) -> None:
    """Local tracing with Logfire: spans and logs of the graph, the agent runs and our own code go to path.

    Nothing is sent to the Logfire service. The console shows notices and above (progress lines),
    the file has everything: graph steps, agent runs, model requests with prompts, answers and token usage,
    and tool calls. Images are left out of the agent spans.
    """
    logfire.configure(
        send_to_logfire=False,
        service_name="vistaqa",
        console=logfire.ConsoleOptions(min_log_level="notice", span_style="simple"),
        additional_span_processors=[SimpleSpanProcessor(_JsonlSpanExporter(path))],
        scrubbing=False,  # local file only; keep prompts and answers verbatim
    )
    logfire.instrument_pydantic_ai(include_binary_content=False)
