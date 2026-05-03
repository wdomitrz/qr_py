#!/usr/bin/env python3
################################################################
# Copyright (c) 2026 Witalis Domitrz <witekdomitrz@gmail.com>
# AGPL License
################################################################

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import cycle
from typing import ClassVar, Literal

Module = Literal["data", "reserved"]
Pixel = bool | None


@dataclass(frozen=True, kw_only=True)
class BitBuffer:
    """Collect big-endian bits.

    >>> buffer = BitBuffer()
    >>> buffer.append(0b0100, 4)
    >>> buffer.append(2, 8)
    >>> buffer.bits[:12]
    [False, True, False, False, False, False, False, False, False, False, True, False]
    >>> buffer.as_codewords()
    [64, 32]
    """

    bits: list[bool] | None = None

    def __post_init__(self) -> None:
        if self.bits is None:
            object.__setattr__(self, "bits", [])

    def append(self, value: int, bit_count: int) -> None:
        assert self.bits is not None
        self.bits.extend(
            bool(value & (1 << shift)) for shift in range(bit_count - 1, -1, -1)
        )

    def pad_to_byte(self) -> None:
        assert self.bits is not None
        while len(self.bits) % 8 != 0:
            self.bits.append(False)

    def as_codewords(self) -> list[int]:
        assert self.bits is not None
        return [
            sum(
                1 << (7 - offset)
                for offset, bit in enumerate(self.bits[index : index + 8])
                if bit
            )
            for index in range(0, len(self.bits), 8)
        ]


class ReedSolomon:
    """QR-compatible Reed-Solomon helpers over GF(256).

    >>> ReedSolomon.multiply(0x57, 0x83)
    49
    >>> ReedSolomon.generator(3)
    [1, 7, 14, 8]
    >>> ReedSolomon.remainder([32, 91, 11, 120, 209, 114, 220], 7)
    [184, 72, 218, 93, 177, 173, 81]
    """

    PRIMITIVE: ClassVar[int] = 0x11D
    FIELD_SIZE: ClassVar[int] = 256

    @staticmethod
    def multiply(left: int, right: int) -> int:
        product = 0
        while right:
            if right & 1:
                product ^= left
            left <<= 1
            if left & ReedSolomon.FIELD_SIZE:
                left ^= ReedSolomon.PRIMITIVE
            right >>= 1
        return product

    @staticmethod
    def generator(degree: int) -> list[int]:
        coefficients = [1]
        root = 1
        for _ in range(degree):
            next_coefficients = [0] * (len(coefficients) + 1)
            for index, coefficient in enumerate(coefficients):
                next_coefficients[index] ^= coefficient
                next_coefficients[index + 1] ^= ReedSolomon.multiply(coefficient, root)
            coefficients = next_coefficients
            root = ReedSolomon.multiply(root, 2)
        return coefficients

    @staticmethod
    def remainder(data: list[int], degree: int) -> list[int]:
        generator = ReedSolomon.generator(degree)
        result = [*data, *([0] * degree)]
        for index, value in enumerate(data):
            if value == 0:
                continue
            for offset, coefficient in enumerate(generator):
                result[index + offset] ^= ReedSolomon.multiply(value, coefficient)
        return result[-degree:]


@dataclass(frozen=True, kw_only=True)
class QRCode:
    """A compact QR Code model for short byte-mode messages.

    >>> code = QRCode.from_text("hello")
    >>> code.size
    21
    >>> code.data_codewords[:6]
    [64, 86, 134, 86, 198, 198]
    >>> len(code.matrix) == code.size and len(code.matrix[0]) == code.size
    True
    >>> rendered = code.render(quiet_zone=1)
    >>> rendered.count("\\n")
    22
    >>> QRCode.from_text("x" * 18)
    Traceback (most recent call last):
    ...
    ValueError: version 1-L QR codes support at most 17 UTF-8 bytes
    """

    text: str
    data_codewords: list[int]
    error_codewords: list[int]
    matrix: list[list[bool]]

    ERROR_CORRECTION: ClassVar[str] = "L"
    VERSION: ClassVar[int] = 1
    SIZE: ClassVar[int] = 21
    DATA_CODEWORDS: ClassVar[int] = 19
    ERROR_CODEWORDS: ClassVar[int] = 7
    BYTE_CAPACITY: ClassVar[int] = 17
    FORMAT_BITS: ClassVar[int] = 0b111011111000100

    @property
    def size(self) -> int:
        return len(self.matrix)

    @classmethod
    def from_text(cls, text: str) -> QRCode:
        data = text.encode()
        if len(data) > cls.BYTE_CAPACITY:
            msg = (
                f"version 1-L QR codes support at most {cls.BYTE_CAPACITY} UTF-8 bytes"
            )
            raise ValueError(msg)
        data_codewords = cls.encode_data(data)
        error_codewords = ReedSolomon.remainder(data_codewords, cls.ERROR_CODEWORDS)
        matrix = QRMatrix.from_codewords([*data_codewords, *error_codewords])
        return cls(
            text=text,
            data_codewords=data_codewords,
            error_codewords=error_codewords,
            matrix=matrix,
        )

    @classmethod
    def encode_data(cls, data: bytes) -> list[int]:
        """Encode bytes in QR byte mode.

        >>> QRCode.encode_data(b"A")[:4]
        [64, 20, 16, 236]
        >>> len(QRCode.encode_data(b"hello"))
        19
        """

        buffer = BitBuffer()
        buffer.append(0b0100, 4)
        buffer.append(len(data), 8)
        for byte in data:
            buffer.append(byte, 8)

        remaining = (cls.DATA_CODEWORDS * 8) - len(buffer.bits or [])
        buffer.append(0, min(4, remaining))
        buffer.pad_to_byte()

        codewords = buffer.as_codewords()
        for pad in cycle((0xEC, 0x11)):
            if len(codewords) == cls.DATA_CODEWORDS:
                return codewords
            codewords.append(pad)
        raise AssertionError

    def render(
        self, *, quiet_zone: int = 2, black: str = "██", white: str = "  "
    ) -> str:
        """Render the QR matrix as terminal text.

        >>> small = QRCode.from_text("A").render(quiet_zone=0, black="1", white="0")
        >>> small.splitlines()[0]
        '111111110010111111111'
            >>> len(small.splitlines())
            21
        """

        blank = white * (self.size + (quiet_zone * 2))
        rows = [blank for _ in range(quiet_zone)]
        rows.extend(
            (
                (white * quiet_zone)
                + "".join(black if cell else white for cell in row)
                + (white * quiet_zone)
            )
            for row in self.matrix
        )
        rows.extend(blank for _ in range(quiet_zone))
        return "\n".join(rows)


class QRMatrix:
    """Matrix construction helpers for version 1 QR codes."""

    FINDER_BORDER: ClassVar[frozenset[int]] = frozenset({0, 6})
    FINDER_CENTER_MIN: ClassVar[int] = 2
    FINDER_CENTER_MAX: ClassVar[int] = 4
    TIMING_ROW_COL: ClassVar[int] = 6

    @staticmethod
    def from_codewords(codewords: list[int]) -> list[list[bool]]:
        modules: list[list[Pixel]] = [
            [None for _ in range(QRCode.SIZE)] for _ in range(QRCode.SIZE)
        ]
        kinds: list[list[Module | None]] = [
            [None for _ in range(QRCode.SIZE)] for _ in range(QRCode.SIZE)
        ]
        QRMatrix.add_patterns(modules, kinds)
        QRMatrix.add_data(modules, kinds, QRMatrix.codeword_bits(codewords))
        QRMatrix.add_format(modules, kinds)
        return [[bool(cell) for cell in row] for row in modules]

    @staticmethod
    def codeword_bits(codewords: list[int]) -> list[bool]:
        return [
            bool(codeword & (1 << shift))
            for codeword in codewords
            for shift in range(7, -1, -1)
        ]

    @staticmethod
    def add_patterns(
        modules: list[list[Pixel]], kinds: list[list[Module | None]]
    ) -> None:
        for row, col in ((0, 0), (0, 14), (14, 0)):
            QRMatrix.add_finder(modules, kinds, row, col)
        for index in range(8, 13):
            QRMatrix.set_reserved(
                modules,
                kinds,
                QRMatrix.TIMING_ROW_COL,
                index,
                value=index % 2 == 0,
            )
            QRMatrix.set_reserved(
                modules,
                kinds,
                index,
                QRMatrix.TIMING_ROW_COL,
                value=index % 2 == 0,
            )
        QRMatrix.set_reserved(modules, kinds, 13, 8, value=True)

    @staticmethod
    def add_finder(
        modules: list[list[Pixel]], kinds: list[list[Module | None]], row: int, col: int
    ) -> None:
        for y in range(-1, 8):
            for x in range(-1, 8):
                current_row = row + y
                current_col = col + x
                if not QRMatrix.in_bounds(current_row, current_col):
                    continue
                is_border = x in QRMatrix.FINDER_BORDER or y in QRMatrix.FINDER_BORDER
                is_center = (
                    QRMatrix.FINDER_CENTER_MIN <= x <= QRMatrix.FINDER_CENTER_MAX
                    and QRMatrix.FINDER_CENTER_MIN <= y <= QRMatrix.FINDER_CENTER_MAX
                )
                QRMatrix.set_reserved(
                    modules,
                    kinds,
                    current_row,
                    current_col,
                    value=is_border or is_center,
                )

    @staticmethod
    def add_data(
        modules: list[list[Pixel]], kinds: list[list[Module | None]], bits: list[bool]
    ) -> None:
        bit_index = 0
        upward = True
        for right_col in range(QRCode.SIZE - 1, 0, -2):
            if right_col == QRMatrix.TIMING_ROW_COL:
                continue
            row_order = range(QRCode.SIZE - 1, -1, -1) if upward else range(QRCode.SIZE)
            for row in row_order:
                for col in (right_col, right_col - 1):
                    if kinds[row][col] is not None:
                        continue
                    bit = bits[bit_index] if bit_index < len(bits) else False
                    modules[row][col] = bit ^ QRMatrix.mask(row, col)
                    kinds[row][col] = "data"
                    bit_index += 1
            upward = not upward

    @staticmethod
    def add_format(
        modules: list[list[Pixel]], kinds: list[list[Module | None]]
    ) -> None:
        positions = [
            (8, 0),
            (8, 1),
            (8, 2),
            (8, 3),
            (8, 4),
            (8, 5),
            (8, 7),
            (8, 8),
            (7, 8),
            (5, 8),
            (4, 8),
            (3, 8),
            (2, 8),
            (1, 8),
            (0, 8),
        ]
        mirror_positions = [
            (20, 8),
            (19, 8),
            (18, 8),
            (17, 8),
            (16, 8),
            (15, 8),
            (14, 8),
            (8, 13),
            (8, 14),
            (8, 15),
            (8, 16),
            (8, 17),
            (8, 18),
            (8, 19),
            (8, 20),
        ]
        bits = [bool(QRCode.FORMAT_BITS & (1 << shift)) for shift in range(14, -1, -1)]
        for bit, (row, col), (mirror_row, mirror_col) in zip(
            bits, positions, mirror_positions, strict=True
        ):
            QRMatrix.set_reserved(modules, kinds, row, col, value=bit)
            QRMatrix.set_reserved(modules, kinds, mirror_row, mirror_col, value=bit)

    @staticmethod
    def set_reserved(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        row: int,
        col: int,
        *,
        value: bool,
    ) -> None:
        modules[row][col] = value
        kinds[row][col] = "reserved"

    @staticmethod
    def in_bounds(row: int, col: int) -> bool:
        return 0 <= row < QRCode.SIZE and 0 <= col < QRCode.SIZE

    @staticmethod
    def mask(row: int, col: int) -> bool:
        return (row + col) % 2 == 0


@dataclass(frozen=True, kw_only=True)
class Args:
    """CLI adapter.

    >>> from contextlib import redirect_stdout
    >>> from io import StringIO
    >>> out = StringIO()
    >>> with redirect_stdout(out):
    ...     exit_code = Args.from_argv(["A", "--quiet-zone", "0"]).main()
    >>> exit_code
    0
    >>> out.getvalue().splitlines()[0]
    '████████████████    ██  ██████████████████'
    """

    text: str
    quiet_zone: int = 2

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> Args:
        parser = argparse.ArgumentParser(description="Generate a terminal QR code.")
        parser.add_argument("text")
        parser.add_argument("--quiet-zone", type=int, default=2)
        namespace = parser.parse_args(argv)
        return cls(text=namespace.text, quiet_zone=namespace.quiet_zone)

    def main(self) -> int:
        print(QRCode.from_text(self.text).render(quiet_zone=self.quiet_zone))
        return 0


if __name__ == "__main__":
    raise SystemExit(Args.from_argv().main())
