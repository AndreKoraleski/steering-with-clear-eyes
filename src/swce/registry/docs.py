"""Rendering of the registry as the benchmark page of the documentation."""

from pathlib import Path

from swce.registry.schema import BenchmarkCard, Registry

DOCS_PATH = Path(__file__).parents[3] / "docs" / "benchmarks.md"
"""Page generated from the registry."""


def _card(card: BenchmarkCard) -> list[str]:
    """Lines documenting one benchmark."""
    lines = [f"## {card.name}", "", f"Key `{card.key}`. {card.reference}.", ""]
    lines += [f"- **Source:** <{card.source}>"]
    if card.revision:
        lines += [f"- **Revision:** `{card.revision}`"]
    lines += [f"- **Licence:** {card.licence}", f"- **Terms:** {card.terms}"]
    if card.image_url:
        lines += [f"- **Images:** fetched one by one from `{card.image_url}`"]
    if card.files:
        lines += ["", "| File | Bytes | Digest |", "|---|---|---|"]
        lines += [
            f"| `{file.path}` | {file.size} | {file.algorithm} `{file.digest}` |"
            for file in card.files
        ]
    if card.tasks:
        lines += ["", "| Task | Kind | Primary metric | Measures |", "|---|---|---|---|"]
        lines += [
            f"| `{task.key}` | {task.kind.value} | `{task.primary}` | {task.description} |"
            for task in card.tasks
        ]
    return [*lines, ""]


def render(registry: Registry) -> str:
    """Benchmark page of a registry.

    Args:
        registry: Registry to document.

    Returns:
        The page in Markdown, ending with a newline.
    """
    lines = [
        "# Benchmarks",
        "",
        "Generated from the registry by `swce-registry`. Do not edit by hand.",
        "",
        f"Registry version {registry.version}. Every file is pinned by its size and digest, and",
        "an evaluation fails when a host serves other bytes.",
        "",
    ]
    for card in registry.benchmarks:
        lines += _card(card)
    return "\n".join(lines).rstrip("\n") + "\n"


def write_docs(registry: Registry, path: Path = DOCS_PATH) -> bool:
    """Write the benchmark page when it is missing or stale.

    Args:
        registry: Registry to document.
        path: Page to write.

    Returns:
        Whether the page was written.
    """
    page = render(registry)
    if path.exists() and path.read_text(encoding="utf-8") == page:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8", newline="\n")
    return True
