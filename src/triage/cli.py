"""CLI. `triage ask` for one-shot, bare `triage` for an interactive loop."""
from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .agent import TriageAgent, build_agent
from .config import get_settings
from .guardrails.contradiction import detect_all_conflicts
from .llm.base import LLMError
from .logging_setup import setup_logging
from .models import TriageResult

console = Console(stderr=False)

ACTION_STYLE = {
    "amend": "green",
    "firm_planned_order": "green",
    "split_child_po": "cyan",
    "raise_backorder": "cyan",
    "escalate": "yellow",
}


def render(result: TriageResult, show_trace: bool = False) -> None:
    rec = result.recommendation
    style = ACTION_STYLE.get(rec.recommended_action, "white")

    body = Table.grid(padding=(0, 2))
    body.add_column(style="dim", justify="right")
    body.add_column()
    body.add_row("PO", rec.po_id)
    body.add_row("Action", f"[bold {style}]{rec.recommended_action}[/]")
    body.add_row("Confidence", rec.confidence)
    body.add_row("Escalate to", rec.escalation_target_role or "[dim]—[/]")
    body.add_row("Citations", escape("\n".join(rec.citations)) or "[red]none[/]")
    body.add_row("Rationale", escape(rec.rationale))
    console.print(Panel(body, title="Triage recommendation", border_style=style))

    if result.guardrail_flags:
        table = Table(title="Guardrails", show_lines=False, header_style="dim")
        table.add_column("")
        table.add_column("flag")
        table.add_column("detail", overflow="fold")
        for flag in result.guardrail_flags:
            mark = "[red]BLOCK[/]" if flag.blocking else "[dim]info[/]"
            table.add_row(mark, flag.name, escape(flag.detail))
        console.print(table)

    console.print(
        f"[dim]provider={result.provider}  steps={result.steps_used}  "
        f"grounding={result.grounding_reason}  "
        f"cosine={result.semantic_similarity if result.semantic_similarity is not None else 'n/a'}  "
        f"tokens={result.usage.get('total_tokens', 0)}[/]"
    )

    if show_trace:
        trace = Table(title="Agent trace", header_style="dim")
        trace.add_column("#", width=3)
        trace.add_column("tool")
        trace.add_column("arguments", overflow="fold")
        trace.add_column("result", overflow="fold")
        for call in result.tool_calls:
            trace.add_row(
                str(call.step),
                call.name,
                escape(json.dumps(call.arguments)[:90]),
                escape(call.error or call.result_summary),
            )
        console.print(trace)


def _agent() -> TriageAgent:
    settings = get_settings()
    console.print(
        f"[dim]provider={settings.triage_llm_provider}  corpus={settings.corpus_dir.name}/[/]"
    )
    return build_agent(settings)


def cmd_ask(args: argparse.Namespace) -> int:
    agent = _agent()
    result = agent.triage(args.question)
    if args.json:
        print(json.dumps(result.model_dump(mode="json"), indent=2))
    else:
        render(result, show_trace=args.trace)
    return 1 if result.blocked and args.strict else 0


def cmd_repl(args: argparse.Namespace) -> int:
    agent = _agent()
    console.print("[dim]Ask about a PO. Ctrl-D or 'exit' to quit.[/]")
    while True:
        try:
            question = console.input("\n[bold cyan]planner>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        try:
            render(agent.triage(question), show_trace=args.trace)
        except LLMError as exc:
            console.print(f"[red]LLM error:[/] {exc}")


def cmd_audit(_: argparse.Namespace) -> int:
    """Corpus hygiene: every threshold inconsistency, independent of any PO."""
    from .retrieval.index import build_index

    settings = get_settings()
    index = build_index(settings, llm=None)  # lexical only; no model needed
    conflicts = detect_all_conflicts(index.chunks)
    console.print(f"[bold]{len(index.chunks)}[/] chunks, retrieval mode: {index.mode}")
    console.print(
        f"PII registry: [bold]{len(index.registry.names)}[/] names, "
        f"[bold]{len(index.registry.emails)}[/] emails redacted at index time"
    )
    if not conflicts:
        console.print("[green]No threshold conflicts detected.[/]")
        return 0
    console.print(f"\n[yellow]{len(conflicts)} threshold conflict(s):[/]")
    for conflict in conflicts:
        console.print(f"  • {escape(conflict.describe())}")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check configuration before a demo, so failures are legible not cryptic."""
    settings = get_settings()
    table = Table(title="Configuration", header_style="dim")
    table.add_column("setting")
    table.add_column("value")
    table.add_row("provider", settings.triage_llm_provider)
    table.add_row("corpus_dir", str(settings.corpus_dir))
    table.add_row("data_dir", str(settings.data_dir))
    table.add_row(
        "semantic retrieval",
        "yes" if settings.semantic_retrieval_configured
        else "[yellow]NO — grounding guardrail will be INACTIVE[/]",
    )
    if settings.triage_llm_provider == "azure":
        table.add_row("azure endpoint", settings.azure_openai_endpoint or "[red]unset[/]")
        table.add_row("azure api key", "set" if settings.azure_openai_api_key else "[red]unset[/]")
        table.add_row("chat deployment", settings.azure_openai_chat_deployment)
        table.add_row(
            "embedding deployment",
            settings.azure_openai_embedding_deployment or "[yellow]unset (lexical-only)[/]",
        )
    console.print(table)

    from .llm.factory import build_llm_client

    try:
        client = build_llm_client(settings)
    except LLMError as exc:
        console.print(f"[red]cannot build client:[/] {exc}")
        return 1

    failed = False
    try:
        reply = client.chat([{"role": "user", "content": "Reply with the single word: ok"}])
        console.print(f"[green]✓ chat deployment reachable[/] ({client.name}): {(reply.text or '').strip()[:40]}")
    except LLMError as exc:
        console.print(f"[red]✗ chat deployment unreachable:[/] {exc}")
        failed = True

    # Probed separately: the embedding deployment is the one most often missing,
    # and without it retrieval drops to lexical-only AND the grounding guardrail
    # goes inactive. A green chat check alone would hide both.
    try:
        vector = client.embed(["connectivity probe"])[0]
        console.print(f"[green]✓ embedding deployment reachable[/] ({len(vector)} dims)")
    except LLMError as exc:
        console.print(f"[yellow]✗ embedding deployment unavailable:[/] {exc}")
        console.print(
            "[yellow]  → retrieval will run lexical-only and the grounding "
            "guardrail will be INACTIVE.[/]"
        )
        failed = True
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description="PO Exception Triage Agent")
    parser.add_argument("--log-level", default=None)
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="Triage a single question")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="Emit the full result envelope")
    ask.add_argument("--trace", action="store_true", help="Show the agent trace")
    ask.add_argument("--strict", action="store_true", help="Exit non-zero if a guardrail blocked")
    ask.set_defaults(func=cmd_ask)

    repl = sub.add_parser("repl", help="Interactive planner loop")
    repl.add_argument("--trace", action="store_true")
    repl.set_defaults(func=cmd_repl)

    sub.add_parser("audit", help="Report corpus threshold conflicts").set_defaults(func=cmd_audit)
    sub.add_parser("doctor", help="Check provider configuration").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    setup_logging(args.log_level or get_settings().log_level)

    if not getattr(args, "func", None):
        args.trace = False
        return cmd_repl(args)
    try:
        return args.func(args)
    except LLMError as exc:
        console.print(f"[red]LLM error:[/] {exc}")
        return 2
    except FileNotFoundError as exc:
        console.print(f"[red]Missing file:[/] {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
