# qr

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

By default, input is read from `stdin`. Use `--text` to pass the payload as an argument.

Long input is split into multiple QR codes by default. Use `--no-split` to fail instead.
