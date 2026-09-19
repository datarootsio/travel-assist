"""The `travel-assist` command.

Subcommands are registered here as they land: `doctor`, `search` and
`chat`.
"""

from __future__ import annotations

import logging

import typer

from travel_assist.cli.chat import chat
from travel_assist.cli.doctor import doctor
from travel_assist.cli.search import search

app = typer.Typer(
    name="travel-assist",
    help="A RAG travel assistant over a Wikivoyage corpus.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Keep `travel-assist` a command group.

    Without a callback, Typer collapses a single-command app into a bare
    command and `travel-assist doctor` becomes "unexpected extra argument".
    This stops being cosmetic the moment more than one subcommand exists.
    """


app.command(name="doctor")(doctor)
app.command(name="search")(search)
app.command(name="chat")(chat)


def main() -> None:
    """Entry point: configure logging, then dispatch."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    app()


if __name__ == "__main__":
    main()
