# qr

Generate terminal QR codes from standard input or an explicit `--text` value.

```console
$ echo -n "https://example.com" | python3 qr.py
```

```console
$ python3 qr.py --text "HELLO WORLD" --error-correction H --mode alphanumeric
```

## CLI

```console
$ python3 qr.py --help
usage: qr.py [-h] [--text TEXT] [--error-correction {L,M,Q,H}]
             [--mode {auto,numeric,alphanumeric,byte,kanji}]
             [--quiet-zone QUIET_ZONE]

Generate a terminal QR code.

options:
  -h, --help            show this help message and exit
  --text TEXT
  --error-correction {L,M,Q,H}
  --mode {auto,numeric,alphanumeric,byte,kanji}
  --quiet-zone QUIET_ZONE
```

By default, input is read from `stdin`. Use `--text` when passing the payload as an argument.

The default quiet zone is `4` modules, which is the standard margin expected by QR scanners. You can override it with `--quiet-zone`, but smaller values can make phone cameras fail to recognize the code.

Terminal output is optimized for dark terminal themes by default: QR dark modules render as spaces, while light modules and the quiet zone render as `██`. This makes the required white quiet-zone border visible to phone cameras.

## Encoding

Supported QR versions: `1` through `40`.

Supported error correction levels:

- `L`: low, largest capacity
- `M`: medium
- `Q`: quartile
- `H`: high, strongest correction and smallest capacity

Supported modes:

- `byte`: UTF-8 bytes; this is the default for text input
- `auto`: choose numeric, alphanumeric, kanji, or byte mode based on the text
- `numeric`: digits only
- `alphanumeric`: uppercase QR alphanumeric character set
- `kanji`: Shift JIS QR kanji ranges

The implementation selects the smallest QR version that can hold the payload for the requested mode and error correction level.

## Development

Run the full local check:

```console
$ make all
```

This runs Ruff fixes/formatting, Ruff linting, BasedPyright, and doctests.
