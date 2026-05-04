# qr_py

Generate QR codes in the terminal.

```console
$ echo -n "https://example.com" | ./qr.py
```

```console
$ ./qr.py --text "HELLO WORLD" --error-correction H --mode alphanumeric
```

```console
$ printf "wifi-password" | ./qr.py wifi --ssid "Home WiFi"
```

```console
$ ./qr.py --text "https://example.com" --format svg --output qr
```

By default, input is read from `stdin`. Use `--text` to pass the payload as an argument.

Long input is split into multiple QR codes by default. Use `--split-mode wait` to advance with Enter, or `--split-mode disabled` to fail instead.

Use `--version 1..40` to force every generated QR code to that version.
