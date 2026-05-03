#!/usr/bin/env python3
################################################################
# Copyright (c) 2026 Witalis Domitrz <witekdomitrz@gmail.com>
# AGPL License
################################################################
#
# /// script
# dependencies = [
#   "typer",
# ]
# ///

from __future__ import annotations

from dataclasses import dataclass

import typer


@dataclass(frozen=True, kw_only=True)
class Args:
    who: str = "World"

    def __post_init__(self) -> None:
        raise typer.Exit(self.main())

    def main(self) -> int:
        print(f"Hello {self.who}!")
        return 0


if __name__ == "__main__":
    typer.run(Args)
