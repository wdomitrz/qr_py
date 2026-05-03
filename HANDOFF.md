# Handoff

## Current State

`qr.py` is a single-file terminal QR generator.

Implemented:

- Reads payload from `stdin` by default.
- Supports `--text` for explicit argument input.
- Supports QR versions `1` through `40`.
- Supports error correction levels `L`, `M`, `Q`, and `H`.
- Supports modes `auto`, `numeric`, `alphanumeric`, `byte`, and `kanji`.
- Defaults text input to `byte` mode; `auto` remains available via `--mode auto`.
- Uses a scanner-friendly default quiet zone of `4` modules.
- Renders terminal QR output for dark terminal themes by default: QR dark modules are spaces, light modules and the quiet zone are `██`.
- Keeps tests inline as doctests in `qr.py`.

`README.md` documents current usage and development commands.

## Verification

Last meaningful validation:

```console
$ make all
ruff check --extend-select I --fix-only --fix .
ruff format .
ruff check .
All checks passed!
basedpyright --project pyproject.toml --level error .
0 errors, 0 warnings, 0 notes
python3 -m doctest README.md qr.py
```

## Notes

- `C408` is intentionally ignored in `pyproject.toml`.
- The QR implementation uses fixed mask pattern `0`.
- Recent correctness fixes adjusted data placement around the timing column and format-bit placement/orientation.
- Finder markers now reserve a full light separator row/column around the 7x7 marker; earlier code accidentally extended black border modules into the separator.
- The default renderer intentionally makes the quiet zone visible as `██`; rendering it as plain spaces on a dark terminal produces an inverted border that phone cameras may reject.
- The code has no runtime third-party dependencies.
- `.codex` is an existing untracked empty file and was not touched.
