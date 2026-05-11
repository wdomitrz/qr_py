#!/usr/bin/env python3
################################################################
# Copyright (c) 2026 Witalis Domitrz <witekdomitrz@gmail.com>
# AGPL License
################################################################

from __future__ import annotations

import argparse
import base64
import binascii
import getpass
import os
import re
import struct
import sys
import zlib
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import ClassVar, Literal, NoReturn, cast


def assert_never(arg: NoReturn) -> NoReturn:
    raise AssertionError(arg)


Module = Literal["data", "reserved"]
Pixel = bool | None
ErrorCorrection = Literal["L", "M", "Q", "H"]
SegmentMode = Literal["numeric", "alphanumeric", "byte", "kanji"]
RequestedMode = Literal["auto", "numeric", "alphanumeric", "byte", "kanji"]
CharacterCountVersionGroup = Literal["1-9", "10-26", "27-40"]
OutputFormat = Literal[
    "terminal", "bits", "ascii", "svg", "html", "bmp", "png", "terminal_img"
]
TerminalImageProtocol = Literal["auto", "kitty", "iterm2"]
SplitMode = Literal["all", "wait", "disabled"]
Command = Literal["text", "wifi"]
WifiAuth = Literal["WPA", "WEP", "nopass"]
ANSI_BLACK = "\033[40m  \033[0m"
ANSI_WHITE = "\033[47m  \033[0m"


@dataclass(kw_only=True)
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

    bits: list[bool] = field(default_factory=list)

    def append(self, value: int, bit_count: int) -> None:
        self.bits.extend(
            bool(value & (1 << shift)) for shift in range(bit_count - 1, -1, -1)
        )

    def pad_to_byte(self) -> None:
        while len(self.bits) % 8 != 0:
            self.bits.append(False)

    def as_codewords(self) -> list[int]:
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
    [7, 14, 8]
    >>> ReedSolomon.remainder([32, 91, 11, 120, 209, 114, 220], 7)
    [255, 198, 226, 122, 164, 250, 136]
    >>> ReedSolomon.remainder([64, 198, 23, 54, 70, 102, 23, 54, 70, 102, 23, 54, 70, 96, 236, 17, 236, 17, 236], 7)
    [221, 150, 217, 99, 43, 41, 158]
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
        coefficients = [0] * (degree - 1) + [1]
        root = 1
        for _ in range(degree):
            for index in range(degree):
                coefficients[index] = ReedSolomon.multiply(coefficients[index], root)
                if index + 1 < degree:
                    coefficients[index] ^= coefficients[index + 1]
            root = ReedSolomon.multiply(root, 2)
        return coefficients

    @staticmethod
    def remainder(data: list[int], degree: int) -> list[int]:
        generator = ReedSolomon.generator(degree)
        result = [0] * degree
        for value in data:
            factor = value ^ result.pop(0)
            result.append(0)
            for index, coefficient in enumerate(generator):
                result[index] ^= ReedSolomon.multiply(coefficient, factor)
        return result


@dataclass(frozen=True, kw_only=True)
class QRVersion:
    version: int
    error_correction: ErrorCorrection
    error_codewords: int
    data_blocks: list[int]
    alignment_positions: list[int]

    SMALL_BYTE_COUNT_MAX_VERSION: ClassVar[int] = 9
    NUMERIC_REMAINDER_TWO_DIGITS: ClassVar[int] = 7
    NUMERIC_REMAINDER_ONE_DIGIT: ClassVar[int] = 4
    ALPHANUMERIC_REMAINDER_ONE_CHAR: ClassVar[int] = 6

    @property
    def size(self) -> int:
        return 17 + (self.version * 4)

    @property
    def data_codewords(self) -> int:
        return sum(self.data_blocks)

    def capacity(self, mode: SegmentMode) -> int:
        """Return the largest payload length for a single segment.

        >>> QRVersions.for_version(1, "L").capacity("numeric")
        41
        >>> QRVersions.for_version(1, "L").capacity("alphanumeric")
        25
        >>> QRVersions.for_version(1, "L").capacity("byte")
        17
        >>> QRVersions.for_version(1, "L").capacity("kanji")
        10
        """

        bits = self.data_codewords * 8
        count_bits = QRSegment.char_count_bits(mode, self.version)
        available = bits - 4 - count_bits
        match mode:
            case "numeric":
                groups, remainder = divmod(available, 10)
                if remainder >= self.NUMERIC_REMAINDER_TWO_DIGITS:
                    return (groups * 3) + 2
                elif remainder >= self.NUMERIC_REMAINDER_ONE_DIGIT:
                    return (groups * 3) + 1
                else:
                    return groups * 3
            case "alphanumeric":
                pairs, remainder = divmod(available, 11)
                return (pairs * 2) + (
                    1 if remainder >= self.ALPHANUMERIC_REMAINDER_ONE_CHAR else 0
                )
            case "byte":
                return available // 8
            case "kanji":
                return available // 13


class QRVersions:
    """QR version metadata for all error correction levels.

    >>> QRVersions.for_text("1" * 41, mode="numeric").version
    1
    >>> QRVersions.for_text("1" * 42, mode="numeric").version
    2
    >>> QRVersions.for_version(40).size
    177
    >>> QRVersions.for_version(40, "L").capacity("byte")
    2953
    >>> QRVersions.for_version(40, "H").capacity("byte")
    1273
    >>> [QRVersions.for_version(version, "L").size for version in range(1, 41)] == [
    ...     17 + (version * 4) for version in range(1, 41)
    ... ]
    True
    >>> capacities = [
    ...     QRVersions.for_version(version, "L").capacity("byte")
    ...     for version in range(1, 41)
    ... ]
    >>> all(left < right for left, right in zip(capacities, capacities[1:]))
    True
    >>> QRVersions.for_text("x" * 2954, error_correction="L", mode="byte")
    Traceback (most recent call last):
    ...
    ValueError: version 40-L byte QR codes support at most 2953 UTF-8 bytes
    """

    MIN_VERSION: ClassVar[int] = 1
    MAX_VERSION: ClassVar[int] = 40
    MIN_ALIGNMENT_VERSION: ClassVar[int] = 2
    MIN_VERSION_INFO_VERSION: ClassVar[int] = 7
    FORMAT_ERROR_CORRECTION_BITS: ClassVar[dict[ErrorCorrection, int]] = {
        "L": 0b01,
        "M": 0b00,
        "Q": 0b11,
        "H": 0b10,
    }
    ECC_CODEWORDS_PER_BLOCK: ClassVar[dict[ErrorCorrection, tuple[int, ...]]] = {
        "L": (
            -1,
            7,
            10,
            15,
            20,
            26,
            18,
            20,
            24,
            30,
            18,
            20,
            24,
            26,
            30,
            22,
            24,
            28,
            30,
            28,
            28,
            28,
            28,
            30,
            30,
            26,
            28,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
        ),
        "M": (
            -1,
            10,
            16,
            26,
            18,
            24,
            16,
            18,
            22,
            22,
            26,
            30,
            22,
            22,
            24,
            24,
            28,
            28,
            26,
            26,
            26,
            26,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
            28,
        ),
        "Q": (
            -1,
            13,
            22,
            18,
            26,
            18,
            24,
            18,
            22,
            20,
            24,
            28,
            26,
            24,
            20,
            30,
            24,
            28,
            28,
            26,
            30,
            28,
            30,
            30,
            30,
            30,
            28,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
        ),
        "H": (
            -1,
            17,
            28,
            22,
            16,
            22,
            28,
            26,
            26,
            24,
            28,
            24,
            28,
            22,
            24,
            24,
            30,
            28,
            28,
            26,
            28,
            30,
            24,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
            30,
        ),
    }
    ERROR_BLOCKS: ClassVar[dict[ErrorCorrection, tuple[int, ...]]] = {
        "L": (
            -1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            4,
            4,
            4,
            4,
            4,
            6,
            6,
            6,
            6,
            7,
            8,
            8,
            9,
            9,
            10,
            12,
            12,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            19,
            20,
            21,
            22,
            24,
            25,
        ),
        "M": (
            -1,
            1,
            1,
            1,
            2,
            2,
            4,
            4,
            4,
            5,
            5,
            5,
            8,
            9,
            9,
            10,
            10,
            11,
            13,
            14,
            16,
            17,
            17,
            18,
            20,
            21,
            23,
            25,
            26,
            28,
            29,
            31,
            33,
            35,
            37,
            38,
            40,
            43,
            45,
            47,
            49,
        ),
        "Q": (
            -1,
            1,
            1,
            2,
            2,
            4,
            4,
            6,
            6,
            8,
            8,
            8,
            10,
            12,
            16,
            12,
            17,
            16,
            18,
            21,
            20,
            23,
            23,
            25,
            27,
            29,
            34,
            34,
            35,
            38,
            40,
            43,
            45,
            48,
            51,
            53,
            56,
            59,
            62,
            65,
            68,
        ),
        "H": (
            -1,
            1,
            1,
            2,
            4,
            4,
            4,
            5,
            6,
            8,
            8,
            11,
            11,
            16,
            16,
            18,
            16,
            19,
            21,
            25,
            25,
            25,
            34,
            30,
            32,
            35,
            37,
            40,
            42,
            45,
            48,
            51,
            54,
            57,
            60,
            63,
            66,
            70,
            74,
            77,
            81,
        ),
    }
    ALIGNMENT_POSITIONS: ClassVar[dict[int, tuple[int, ...]]] = {
        1: (),
        2: (6, 18),
        3: (6, 22),
        4: (6, 26),
        5: (6, 30),
        6: (6, 34),
        7: (6, 22, 38),
        8: (6, 24, 42),
        9: (6, 26, 46),
        10: (6, 28, 50),
        11: (6, 30, 54),
        12: (6, 32, 58),
        13: (6, 34, 62),
        14: (6, 26, 46, 66),
        15: (6, 26, 48, 70),
        16: (6, 26, 50, 74),
        17: (6, 30, 54, 78),
        18: (6, 30, 56, 82),
        19: (6, 30, 58, 86),
        20: (6, 34, 62, 90),
        21: (6, 28, 50, 72, 94),
        22: (6, 26, 50, 74, 98),
        23: (6, 30, 54, 78, 102),
        24: (6, 28, 54, 80, 106),
        25: (6, 32, 58, 84, 110),
        26: (6, 30, 58, 86, 114),
        27: (6, 34, 62, 90, 118),
        28: (6, 26, 50, 74, 98, 122),
        29: (6, 30, 54, 78, 102, 126),
        30: (6, 26, 52, 78, 104, 130),
        31: (6, 30, 56, 82, 108, 134),
        32: (6, 34, 60, 86, 112, 138),
        33: (6, 30, 58, 86, 114, 142),
        34: (6, 34, 62, 90, 118, 146),
        35: (6, 30, 54, 78, 102, 126, 150),
        36: (6, 24, 50, 76, 102, 128, 154),
        37: (6, 28, 54, 80, 106, 132, 158),
        38: (6, 32, 58, 84, 110, 136, 162),
        39: (6, 26, 54, 82, 110, 138, 166),
        40: (6, 30, 58, 86, 114, 142, 170),
    }

    @classmethod
    def for_version(
        cls, version: int, error_correction: ErrorCorrection = "L"
    ) -> QRVersion:
        if not cls.MIN_VERSION <= version <= cls.MAX_VERSION:
            msg = "QR version must be between 1 and 40"
            raise ValueError(msg)
        raw_codewords = cls.raw_data_modules(version) // 8
        error_codewords = cls.ECC_CODEWORDS_PER_BLOCK[error_correction][version]
        block_count = cls.ERROR_BLOCKS[error_correction][version]
        short_block_count = block_count - (raw_codewords % block_count)
        short_block_length = raw_codewords // block_count
        data_blocks = [
            short_block_length
            - error_codewords
            + (0 if block < short_block_count else 1)
            for block in range(block_count)
        ]
        return QRVersion(
            version=version,
            error_correction=error_correction,
            error_codewords=error_codewords,
            data_blocks=data_blocks,
            alignment_positions=list(cls.ALIGNMENT_POSITIONS[version]),
        )

    @classmethod
    def for_text(
        cls,
        text: str,
        *,
        error_correction: ErrorCorrection = "L",
        mode: RequestedMode = "byte",
    ) -> QRVersion:
        segment = QRSegment.from_text(text, mode=mode)
        return cls.for_segment(segment, error_correction=error_correction)

    @classmethod
    def for_segment(
        cls, segment: QRSegment, *, error_correction: ErrorCorrection = "L"
    ) -> QRVersion:
        for version in range(cls.MIN_VERSION, cls.MAX_VERSION + 1):
            metadata = cls.for_version(version, error_correction)
            used_bits = segment.total_bits(version)
            if used_bits is not None and used_bits <= metadata.data_codewords * 8:
                return metadata
        metadata = cls.for_version(cls.MAX_VERSION, error_correction)
        msg = (
            f"version 40-{error_correction} {segment.mode} QR codes support at most "
            f"{metadata.capacity(segment.mode)} {segment.unit_name}"
        )
        raise ValueError(msg)

    @classmethod
    def raw_data_modules(cls, version: int) -> int:
        if not cls.MIN_VERSION <= version <= cls.MAX_VERSION:
            msg = "QR version must be between 1 and 40"
            raise ValueError(msg)
        result = ((16 * version) + 128) * version + 64
        if version >= cls.MIN_ALIGNMENT_VERSION:
            alignment_count = (version // 7) + 2
            result -= ((25 * alignment_count) - 10) * alignment_count - 55
        if version >= cls.MIN_VERSION_INFO_VERSION:
            result -= 36
        return result


@dataclass(frozen=True, kw_only=True)
class QRSegment:
    """A single QR data segment.

    >>> QRSegment.from_text("12345").mode
    'numeric'
    >>> QRSegment.from_text("HELLO WORLD").mode
    'alphanumeric'
    >>> QRSegment.from_text("hello").mode
    'byte'
    >>> QRSegment.from_text("漢字").mode
    'kanji'
    >>> QRSegment.from_text("123", mode="alphanumeric").mode
    'alphanumeric'
    >>> QRSegment.from_text("abc", mode="numeric")
    Traceback (most recent call last):
    ...
    ValueError: text cannot be encoded in numeric mode
    """

    mode: SegmentMode
    character_count: int
    bits: list[bool]

    ALPHANUMERIC_CHARACTERS: ClassVar[str] = (
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    )
    ALPHANUMERIC_VALUES: ClassVar[dict[str, int]] = {
        char: index for index, char in enumerate(ALPHANUMERIC_CHARACTERS)
    }
    NUMERIC_RE: ClassVar[re.Pattern[str]] = re.compile(r"[0-9]*")
    ALPHANUMERIC_RE: ClassVar[re.Pattern[str]] = re.compile(r"[A-Z0-9 $%*+./:-]*")
    MODE_BITS: ClassVar[dict[SegmentMode, int]] = {
        "numeric": 0x1,
        "alphanumeric": 0x2,
        "byte": 0x4,
        "kanji": 0x8,
    }
    COUNT_BITS: ClassVar[dict[SegmentMode, dict[CharacterCountVersionGroup, int]]] = {
        "numeric": {"1-9": 10, "10-26": 12, "27-40": 14},
        "alphanumeric": {"1-9": 9, "10-26": 11, "27-40": 13},
        "byte": {"1-9": 8, "10-26": 16, "27-40": 16},
        "kanji": {"1-9": 8, "10-26": 10, "27-40": 12},
    }
    SHIFT_JIS_BYTES_PER_KANJI: ClassVar[int] = 2
    KANJI_RANGE_1_MIN: ClassVar[int] = 0x8140
    KANJI_RANGE_1_MAX: ClassVar[int] = 0x9FFC
    KANJI_RANGE_1_OFFSET: ClassVar[int] = 0x8140
    KANJI_RANGE_2_MIN: ClassVar[int] = 0xE040
    KANJI_RANGE_2_MAX: ClassVar[int] = 0xEBBF
    KANJI_RANGE_2_OFFSET: ClassVar[int] = 0xC140
    SMALL_VERSION_MAX: ClassVar[int] = 9
    MEDIUM_VERSION_MIN: ClassVar[int] = 10
    MEDIUM_VERSION_MAX: ClassVar[int] = 26
    LARGE_VERSION_MIN: ClassVar[int] = 27
    LARGE_VERSION_MAX: ClassVar[int] = 40

    @property
    def unit_name(self) -> Literal["UTF-8 bytes", "characters"]:
        return "UTF-8 bytes" if self.mode == "byte" else "characters"

    @classmethod
    def from_text(cls, text: str, *, mode: RequestedMode = "auto") -> QRSegment:
        selected_mode = cls.best_mode(text) if mode == "auto" else mode
        match selected_mode:
            case "numeric":
                return cls.numeric(text)
            case "alphanumeric":
                return cls.alphanumeric(text)
            case "byte":
                return cls.byte(text)
            case "kanji":
                return cls.kanji(text)

    @classmethod
    def best_mode(cls, text: str) -> SegmentMode:
        if cls.NUMERIC_RE.fullmatch(text) is not None:
            return "numeric"
        elif cls.ALPHANUMERIC_RE.fullmatch(text) is not None:
            return "alphanumeric"
        elif cls.can_encode_kanji(text):
            return "kanji"
        else:
            return "byte"

    @classmethod
    def numeric(cls, text: str) -> QRSegment:
        if cls.NUMERIC_RE.fullmatch(text) is None:
            msg = "text cannot be encoded in numeric mode"
            raise ValueError(msg)
        buffer = BitBuffer()
        for index in range(0, len(text), 3):
            chunk = text[index : index + 3]
            buffer.append(int(chunk), (len(chunk) * 3) + 1)
        return cls(mode="numeric", character_count=len(text), bits=buffer.bits)

    @classmethod
    def alphanumeric(cls, text: str) -> QRSegment:
        if cls.ALPHANUMERIC_RE.fullmatch(text) is None:
            msg = "text cannot be encoded in alphanumeric mode"
            raise ValueError(msg)
        buffer = BitBuffer()
        for index in range(0, len(text) - 1, 2):
            value = (
                cls.ALPHANUMERIC_VALUES[text[index]] * 45
            ) + cls.ALPHANUMERIC_VALUES[text[index + 1]]
            buffer.append(value, 11)
        if len(text) % 2:
            buffer.append(cls.ALPHANUMERIC_VALUES[text[-1]], 6)
        return cls(mode="alphanumeric", character_count=len(text), bits=buffer.bits)

    @classmethod
    def byte(cls, text: str) -> QRSegment:
        buffer = BitBuffer()
        data = text.encode()
        for byte in data:
            buffer.append(byte, 8)
        return cls(mode="byte", character_count=len(data), bits=buffer.bits)

    @classmethod
    def kanji(cls, text: str) -> QRSegment:
        buffer = BitBuffer()
        for character in text:
            value = cls.kanji_value(character)
            if value is None:
                msg = "text cannot be encoded in kanji mode"
                raise ValueError(msg)
            buffer.append(value, 13)
        return cls(mode="kanji", character_count=len(text), bits=buffer.bits)

    @classmethod
    def can_encode_kanji(cls, text: str) -> bool:
        return all(cls.kanji_value(character) is not None for character in text)

    @staticmethod
    def kanji_value(character: str) -> int | None:
        try:
            encoded = character.encode("shift_jis")
        except UnicodeEncodeError:
            return None
        if len(encoded) != QRSegment.SHIFT_JIS_BYTES_PER_KANJI:
            return None
        value = (encoded[0] << 8) | encoded[1]
        if QRSegment.KANJI_RANGE_1_MIN <= value <= QRSegment.KANJI_RANGE_1_MAX:
            value -= QRSegment.KANJI_RANGE_1_OFFSET
        elif QRSegment.KANJI_RANGE_2_MIN <= value <= QRSegment.KANJI_RANGE_2_MAX:
            value -= QRSegment.KANJI_RANGE_2_OFFSET
        else:
            return None
        return ((value >> 8) * 0xC0) + (value & 0xFF)

    @classmethod
    def char_count_bits(cls, mode: SegmentMode, version: int) -> int:
        return cls.COUNT_BITS[mode][cls.character_count_version_group(version)]

    @staticmethod
    def character_count_version_group(version: int) -> CharacterCountVersionGroup:
        if QRVersions.MIN_VERSION <= version <= QRSegment.SMALL_VERSION_MAX:
            return "1-9"
        elif QRSegment.MEDIUM_VERSION_MIN <= version <= QRSegment.MEDIUM_VERSION_MAX:
            return "10-26"
        elif QRSegment.LARGE_VERSION_MIN <= version <= QRSegment.LARGE_VERSION_MAX:
            return "27-40"
        else:
            msg = "QR version must be between 1 and 40"
            raise ValueError(msg)

    def total_bits(self, version: int) -> int | None:
        count_bits = self.char_count_bits(self.mode, version)
        if self.character_count >= (1 << count_bits):
            return None
        return 4 + count_bits + len(self.bits)


@dataclass(frozen=True, kw_only=True)
class QRCode:
    """A compact QR Code model.

    >>> code = QRCode.from_text("hello")
    >>> code.size
    21
    >>> code.mode
    'byte'
    >>> len(code.matrix) == code.size and len(code.matrix[0]) == code.size
    True
    >>> rendered = QRRenderer.render_terminal(code.rows(), quiet_zone=1)
    >>> rendered.count("\\n")
    22
    >>> QRCode.from_text("1" * 42).version
    2
    >>> QRCode.from_text("1" * 42, mode="byte").version
    3
    >>> all(
    ...     QRCode.from_text("x" * QRVersions.for_version(version, "L").capacity("byte"), mode="byte").version
    ...     == version
    ...     for version in range(1, 41)
    ... )
    True
    >>> all(
    ...     QRCode.from_text("x" * (QRVersions.for_version(version - 1, "L").capacity("byte") + 1), mode="byte").version
    ...     == version
    ...     for version in range(2, 41)
    ... )
    True
    >>> max_code = QRCode.from_text("x" * QRVersions.for_version(40, "H").capacity("byte"), error_correction="H", mode="byte")
    >>> (max_code.version, max_code.error_correction, max_code.size)
    (40, 'H', 177)
    >>> QRCode.from_text("A", version=10).version
    10
    >>> QRCode.from_text("x" * 18, version=1)
    Traceback (most recent call last):
    ...
    ValueError: version 1-L byte QR codes support at most 17 UTF-8 bytes
    >>> QRCode.from_text("1234567890").mode
    'numeric'
    >>> QRCode.from_text("asdfasdf").mode
    'byte'
    >>> QRCode.from_text("1234567890", mode="auto").mode
    'numeric'
    >>> QRCode.from_text("HELLO WORLD", mode="auto").mode
    'alphanumeric'
    >>> QRCode.from_text("漢字", mode="auto").mode
    'kanji'
    >>> samples = {
    ...     "numeric": "1234567890",
    ...     "alphanumeric": "HELLO WORLD",
    ...     "byte": "hello",
    ...     "kanji": "漢字",
    ... }
    >>> all(
    ...     QRCode.from_text(sample, error_correction=level, mode=mode).mode == mode
    ...     and QRCode.from_text(sample, error_correction=level, mode=mode).error_correction == level
    ...     for level in ("L", "M", "Q", "H")
    ...     for mode, sample in samples.items()
    ... )
    True
    >>> QRCode.from_text("asdfasdfasdf", mode="byte").error_codewords
    [221, 150, 217, 99, 43, 41, 158]
    """

    text: str
    version: int
    error_correction: ErrorCorrection
    mode: SegmentMode
    data_codewords: list[int]
    error_codewords: list[int]
    matrix: list[list[bool]]

    DEFAULT_ERROR_CORRECTION: ClassVar[ErrorCorrection] = "L"
    DEFAULT_MODE: ClassVar[RequestedMode] = "auto"

    @property
    def size(self) -> int:
        return len(self.matrix)

    def rows(self) -> list[list[int]]:
        """Return the generated QR modules as an intermediate 0/1 matrix.

        >>> QRCode.from_text("A").rows()[0]
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]
        """

        return [[1 if cell else 0 for cell in row] for row in self.matrix]

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        error_correction: ErrorCorrection = DEFAULT_ERROR_CORRECTION,
        mode: RequestedMode = DEFAULT_MODE,
        version: int | None = None,
    ) -> QRCode:
        segment = QRSegment.from_text(text, mode=mode)
        metadata = (
            QRVersions.for_version(version, error_correction)
            if version is not None
            else QRVersions.for_segment(segment, error_correction=error_correction)
        )
        used_bits = segment.total_bits(metadata.version)
        if used_bits is None or used_bits > metadata.data_codewords * 8:
            msg = (
                f"version {metadata.version}-{error_correction} {segment.mode} QR "
                f"codes support at most {metadata.capacity(segment.mode)} "
                f"{segment.unit_name}"
            )
            raise ValueError(msg)
        data_codewords = cls.encode_segment(segment, metadata)
        blocks = cls.make_blocks(data_codewords, metadata)
        codewords = cls.interleave_blocks(blocks)
        error_codewords = [codeword for _, error in blocks for codeword in error]
        matrix = QRMatrix.from_codewords(codewords, metadata)
        return cls(
            text=text,
            version=metadata.version,
            error_correction=error_correction,
            mode=segment.mode,
            data_codewords=data_codewords,
            error_codewords=error_codewords,
            matrix=matrix,
        )

    @classmethod
    def encode_data(
        cls,
        data: bytes,
        version: QRVersion | None = None,
        *,
        error_correction: ErrorCorrection = DEFAULT_ERROR_CORRECTION,
    ) -> list[int]:
        """Encode raw bytes in QR byte mode.

        >>> QRCode.encode_data(b"A")[:4]
        [64, 20, 16, 236]
        >>> len(QRCode.encode_data(b"hello"))
        19
        >>> metadata = QRVersions.for_version(10, "L")
        >>> codewords = QRCode.encode_data(b"x" * metadata.capacity("byte"), metadata)
        >>> codewords[:3]
        [64, 16, 247]
        >>> len(codewords) == metadata.data_codewords
        True
        """

        buffer = BitBuffer()
        for byte in data:
            buffer.append(byte, 8)
        segment = QRSegment(mode="byte", character_count=len(data), bits=buffer.bits)
        metadata = version or QRVersions.for_segment(
            segment, error_correction=error_correction
        )
        return cls.encode_segment(segment, metadata)

    @staticmethod
    def encode_segment(segment: QRSegment, version: QRVersion) -> list[int]:
        """Encode a segment and pad it to the QR version's data capacity.

        >>> QRCode.encode_segment(QRSegment.from_text("12345"), QRVersions.for_version(1, "L"))[:3]
        [16, 20, 123]
        """

        buffer = BitBuffer()
        buffer.append(QRSegment.MODE_BITS[segment.mode], 4)
        buffer.append(
            segment.character_count,
            QRSegment.char_count_bits(segment.mode, version.version),
        )
        for bit in segment.bits:
            buffer.append(int(bit), 1)

        remaining = (version.data_codewords * 8) - len(buffer.bits)
        buffer.append(0, min(4, remaining))
        buffer.pad_to_byte()

        codewords = buffer.as_codewords()
        for pad in cycle((0xEC, 0x11)):
            if len(codewords) == version.data_codewords:
                return codewords
            codewords.append(pad)
        raise AssertionError

    @staticmethod
    def make_blocks(
        data_codewords: list[int], version: QRVersion
    ) -> list[tuple[list[int], list[int]]]:
        blocks: list[tuple[list[int], list[int]]] = []
        offset = 0
        for block_size in version.data_blocks:
            block = data_codewords[offset : offset + block_size]
            blocks.append(
                (block, ReedSolomon.remainder(block, version.error_codewords))
            )
            offset += block_size
        return blocks

    @staticmethod
    def interleave_blocks(blocks: list[tuple[list[int], list[int]]]) -> list[int]:
        """Interleave data blocks, then error correction blocks.

        >>> QRCode.interleave_blocks([([1, 2], [10, 11]), ([3], [12, 13])])
        [1, 3, 2, 10, 12, 11, 13]
        """

        result: list[int] = []
        for block_kind in ("data", "error"):
            match block_kind:
                case "data":
                    selected = [data for data, _ in blocks]
                case "error":
                    selected = [error for _, error in blocks]
            max_length = max(len(block) for block in selected)
            for index in range(max_length):
                result.extend(block[index] for block in selected if index < len(block))
        return result


class QRRenderer:
    """Render 0/1 QR module matrices into concrete output formats.

    >>> rows = QRCode.from_text("A").rows()
    >>> QRRenderer.render_bits(rows).splitlines()[0]
    '111111100101101111111'
    >>> QRRenderer.render_ascii(rows, quiet_zone=0).splitlines()[0]
    '#######  # ## #######'
    >>> QRRenderer.render_svg(rows, quiet_zone=0).startswith('<svg ')
    True
    >>> '<svg ' in QRRenderer.render_html([rows])
    True
    >>> QRRenderer.render_bmp(rows, quiet_zone=0, scale=1)[:2]
    b'BM'
    >>> QRRenderer.render_png(rows, quiet_zone=0, scale=1)[:8]
    b'\\x89PNG\\r\\n\\x1a\\n'
    >>> QRRenderer.render_terminal_image(rows, quiet_zone=0, scale=1, protocol="kitty").startswith('\\x1b_Ga=T,f=100;')
    True
    >>> QRRenderer.render_terminal_image(rows, quiet_zone=0, scale=1, protocol="kitty").endswith('\\x1b\\\\')
    True
    >>> QRRenderer.render_terminal_image(rows, quiet_zone=0, scale=1, protocol="iterm2").startswith('\\x1b]1337;File=inline=1;size=')
    True
    >>> QRRenderer.terminal_image_protocol({"KITTY_WINDOW_ID": "1"})
    'kitty'
    >>> QRRenderer.terminal_image_protocol({"TERM_PROGRAM": "iTerm.app"})
    'iterm2'
    >>> large_rows = QRCode.from_text("x" * QRVersions.for_version(40, "L").capacity("byte"), mode="byte").rows()
    >>> large_kitty = QRRenderer.render_terminal_image(large_rows, protocol="kitty")
    >>> large_kitty.startswith('\\x1b_Ga=T,f=100,m=1;')
    True
    >>> large_kitty.count('\\x1b_G') > 1 and large_kitty.endswith('\\x1b\\\\')
    True
    """

    BLACK: ClassVar[bytes] = b"\x00\x00\x00"
    WHITE: ClassVar[bytes] = b"\xff\xff\xff"
    KITTY_CHUNK_SIZE: ClassVar[int] = 4096

    @staticmethod
    def with_quiet_zone(rows: list[list[int]], quiet_zone: int) -> list[list[int]]:
        if quiet_zone <= 0:
            return [row[:] for row in rows]
        width = len(rows[0]) + (quiet_zone * 2)
        blank = [0] * width
        return (
            [blank[:] for _ in range(quiet_zone)]
            + [[0] * quiet_zone + row[:] + [0] * quiet_zone for row in rows]
            + [blank[:] for _ in range(quiet_zone)]
        )

    @staticmethod
    def render_terminal(
        rows: list[list[int]],
        *,
        quiet_zone: int = 2,
        black: str = ANSI_BLACK,
        white: str = ANSI_WHITE,
    ) -> str:
        return "\n".join(
            "".join(black if cell else white for cell in row)
            for row in QRRenderer.with_quiet_zone(rows, quiet_zone)
        )

    @staticmethod
    def render_bits(rows: list[list[int]]) -> str:
        return "\n".join("".join(str(cell) for cell in row) for row in rows)

    @staticmethod
    def render_ascii(rows: list[list[int]], *, quiet_zone: int = 2) -> str:
        return "\n".join(
            "".join("#" if cell else " " for cell in row)
            for row in QRRenderer.with_quiet_zone(rows, quiet_zone)
        )

    @staticmethod
    def render_svg(
        rows: list[list[int]], *, quiet_zone: int = 2, scale: int = 10
    ) -> str:
        modules = QRRenderer.with_quiet_zone(rows, quiet_zone)
        size = len(modules)
        rects = [
            f'<rect x="{col * scale}" y="{row * scale}" width="{scale}" height="{scale}"/>'
            for row, line in enumerate(modules)
            for col, cell in enumerate(line)
            if cell
        ]
        pixel_size = size * scale
        return "".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{pixel_size}" ',
                f'height="{pixel_size}" viewBox="0 0 {pixel_size} {pixel_size}" ',
                'shape-rendering="crispEdges">',
                f'<rect width="{pixel_size}" height="{pixel_size}" fill="#fff"/>',
                '<g fill="#000">',
                *rects,
                "</g></svg>",
            ]
        )

    @staticmethod
    def render_html(
        rows_list: list[list[list[int]]], *, quiet_zone: int = 2, scale: int = 10
    ) -> str:
        svgs = "\n".join(
            f"<figure>{QRRenderer.render_svg(rows, quiet_zone=quiet_zone, scale=scale)}</figure>"
            for rows in rows_list
        )
        return (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>QR Code</title>"
            "<style>body{font-family:sans-serif;margin:24px;background:#fff;color:#111}"
            "figure{margin:0 0 24px}svg{max-width:100%;height:auto}</style>"
            "</head><body>"
            f"{svgs}</body></html>"
        )

    @staticmethod
    def render_bmp(
        rows: list[list[int]], *, quiet_zone: int = 2, scale: int = 10
    ) -> bytes:
        pixels = QRRenderer.scaled_pixels(rows, quiet_zone=quiet_zone, scale=scale)
        height = len(pixels)
        width = len(pixels[0])
        row_stride = ((width * 3) + 3) & ~3
        pixel_data = bytearray()
        for row in reversed(pixels):
            data = b"".join(
                QRRenderer.BLACK if cell else QRRenderer.WHITE for cell in row
            )
            pixel_data.extend(data)
            pixel_data.extend(b"\x00" * (row_stride - len(data)))
        header_size = 14 + 40
        file_size = header_size + len(pixel_data)
        return (
            b"BM"
            + struct.pack("<IHHI", file_size, 0, 0, header_size)
            + struct.pack(
                "<IIIHHIIIIII", 40, width, height, 1, 24, 0, len(pixel_data), 0, 0, 0, 0
            )
            + bytes(pixel_data)
        )

    @staticmethod
    def render_png(
        rows: list[list[int]], *, quiet_zone: int = 2, scale: int = 10
    ) -> bytes:
        pixels = QRRenderer.scaled_pixels(rows, quiet_zone=quiet_zone, scale=scale)
        height = len(pixels)
        width = len(pixels[0])
        scanlines = b"".join(
            b"\x00"
            + b"".join(QRRenderer.BLACK if cell else QRRenderer.WHITE for cell in row)
            for row in pixels
        )
        return (
            b"\x89PNG\r\n\x1a\n"
            + QRRenderer.png_chunk(
                b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            )
            + QRRenderer.png_chunk(b"IDAT", zlib.compress(scanlines))
            + QRRenderer.png_chunk(b"IEND", b"")
        )

    @staticmethod
    def render_terminal_image(
        rows: list[list[int]],
        *,
        quiet_zone: int = 2,
        scale: int = 10,
        protocol: TerminalImageProtocol = "auto",
    ) -> str:
        png = QRRenderer.render_png(rows, quiet_zone=quiet_zone, scale=scale)
        match QRRenderer.terminal_image_protocol(os.environ, requested=protocol):
            case "kitty":
                return QRRenderer.render_kitty_png(png)
            case "iterm2":
                return QRRenderer.render_iterm2_png(png)

    @staticmethod
    def terminal_image_protocol(
        env: Mapping[str, str],
        *,
        requested: TerminalImageProtocol = "auto",
    ) -> Literal["kitty", "iterm2"]:
        if requested != "auto":
            return requested
        elif env.get("KITTY_WINDOW_ID") or "kitty" in env.get("TERM", "").lower():
            return "kitty"
        elif env.get("TERM_PROGRAM") == "iTerm.app":
            return "iterm2"
        else:
            return "kitty"

    @staticmethod
    def render_kitty_png(png: bytes) -> str:
        encoded = base64.b64encode(png).decode("ascii")
        chunks = [
            encoded[index : index + QRRenderer.KITTY_CHUNK_SIZE]
            for index in range(0, len(encoded), QRRenderer.KITTY_CHUNK_SIZE)
        ]
        if len(chunks) == 1:
            return f"\033_Ga=T,f=100;{chunks[0]}\033\\"
        packets = [f"\033_Ga=T,f=100,m=1;{chunks[0]}\033\\"]
        packets.extend(f"\033_Gm=1;{chunk}\033\\" for chunk in chunks[1:-1])
        packets.append(f"\033_Gm=0;{chunks[-1]}\033\\")
        return "".join(packets)

    @staticmethod
    def render_iterm2_png(png: bytes) -> str:
        encoded = base64.b64encode(png).decode("ascii")
        return f"\033]1337;File=inline=1;size={len(png)}:{encoded}\a"

    @staticmethod
    def png_chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)
        )

    @staticmethod
    def scaled_pixels(
        rows: list[list[int]], *, quiet_zone: int = 2, scale: int = 10
    ) -> list[list[int]]:
        modules = QRRenderer.with_quiet_zone(rows, quiet_zone)
        return [
            [cell for cell in line for _ in range(scale)]
            for line in modules
            for _ in range(scale)
        ]


class QRMatrix:
    """Matrix construction helpers.

    >>> len(QRMatrix.from_codewords([], QRVersions.for_version(7)))
    45
    >>> QRMatrix.version_bits(7)
    31892
    >>> [QRMatrix.format_bits(level, mask=0) for level in ("L", "M", "Q", "H")]
    [30660, 21522, 13663, 5769]
    >>> code = QRCode.from_text("x" * QRVersions.for_version(40, "L").capacity("byte"), mode="byte")
    >>> size = code.size
    >>> all(
    ...     code.matrix[row][col]
    ...     and code.matrix[row + 6][col + 6]
    ...     and code.matrix[row + 3][col + 3]
    ...     for row, col in ((0, 0), (0, size - 7), (size - 7, 0))
    ... )
    True
    >>> code = QRCode.from_text("x" * QRVersions.for_version(2, "L").capacity("byte"), mode="byte")
    >>> code.matrix[18][18], code.matrix[17][17], code.matrix[16][16]
    (True, False, True)
    >>> matrix = QRMatrix.from_codewords([], QRVersions.for_version(7))
    >>> any(matrix[row][34] for row in range(6))
    True
    >>> any(matrix[34][col] for col in range(6))
    True
    >>> code = QRCode.from_text("separator check")
    >>> all(not code.matrix[7][col] for col in range(8))
    True
    >>> all(not code.matrix[row][7] for row in range(8))
    True
    >>> all(not code.matrix[7][code.size - 8 + col] for col in range(8))
    True
    >>> all(not code.matrix[code.size - 8 + row][7] for row in range(8))
    True
    >>> version = QRVersions.for_version(1, "L")
    >>> modules = [[None for _ in range(version.size)] for _ in range(version.size)]
    >>> kinds = [[None for _ in range(version.size)] for _ in range(version.size)]
    >>> QRMatrix.add_patterns(modules, kinds, version)
    >>> QRMatrix.add_data(modules, kinds, [False] * (QRVersions.raw_data_modules(1) // 8 * 8))
    >>> any(row[0] == "data" for row in kinds)
    True
    >>> import re
    >>> import shutil
    >>> import subprocess
    >>> def qrterminal_matrix(text: str) -> list[str]:
    ...     if shutil.which("qrterminal") is None:
    ...         return []
    ...     output = subprocess.check_output(["qrterminal", "-q", "0", text], text=True)
    ...     rows = []
    ...     for line in output.splitlines():
    ...         cells = re.findall(r"\\x1b\\[(4[07])m  \\x1b\\[0m", line)
    ...         if cells:
    ...             rows.append("".join("1" if cell == "40" else "0" for cell in cells))
    ...     return [row[1:22] for row in rows[1:22]]
    >>> ours = ["".join("1" if cell else "0" for cell in row) for row in QRCode.from_text("asdfasdfasdf", mode="byte").matrix]
    >>> reference = qrterminal_matrix("asdfasdfasdf")
    >>> not reference or ours == reference
    True
    """

    FINDER_BORDER: ClassVar[frozenset[int]] = frozenset({0, 6})
    FINDER_MIN: ClassVar[int] = 0
    FINDER_MAX: ClassVar[int] = 6
    FINDER_CENTER_MIN: ClassVar[int] = 2
    FINDER_CENTER_MAX: ClassVar[int] = 4
    TIMING_ROW_COL: ClassVar[int] = 6
    ALIGNMENT_RADIUS: ClassVar[int] = 2
    VERSION_INFO_MIN_VERSION: ClassVar[int] = 7

    @staticmethod
    def from_codewords(codewords: list[int], version: QRVersion) -> list[list[bool]]:
        modules: list[list[Pixel]] = [
            [None for _ in range(version.size)] for _ in range(version.size)
        ]
        kinds: list[list[Module | None]] = [
            [None for _ in range(version.size)] for _ in range(version.size)
        ]
        QRMatrix.add_patterns(modules, kinds, version)
        QRMatrix.add_data(modules, kinds, QRMatrix.codeword_bits(codewords))
        QRMatrix.add_format(modules, kinds, version)
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
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        version: QRVersion,
    ) -> None:
        size = version.size
        for row, col in ((0, 0), (0, size - 7), (size - 7, 0)):
            QRMatrix.add_finder(modules, kinds, row, col)
        for index in range(8, size - 8):
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
        for row in version.alignment_positions:
            for col in version.alignment_positions:
                is_finder_overlap = (
                    (row == QRMatrix.TIMING_ROW_COL and col == QRMatrix.TIMING_ROW_COL)
                    or (
                        row == QRMatrix.TIMING_ROW_COL
                        and col == version.alignment_positions[-1]
                    )
                    or (
                        row == version.alignment_positions[-1]
                        and col == QRMatrix.TIMING_ROW_COL
                    )
                )
                if not is_finder_overlap:
                    QRMatrix.add_alignment(modules, kinds, row, col)
        QRMatrix.reserve_format(modules, kinds, version)
        if version.version >= QRMatrix.VERSION_INFO_MIN_VERSION:
            QRMatrix.reserve_version(modules, kinds, version)
        QRMatrix.set_reserved(modules, kinds, size - 8, 8, value=True)

    @staticmethod
    def add_finder(
        modules: list[list[Pixel]], kinds: list[list[Module | None]], row: int, col: int
    ) -> None:
        for y in range(-1, 8):
            for x in range(-1, 8):
                current_row = row + y
                current_col = col + x
                if not QRMatrix.in_bounds(current_row, current_col, len(modules)):
                    continue
                is_finder = (
                    QRMatrix.FINDER_MIN <= x <= QRMatrix.FINDER_MAX
                    and QRMatrix.FINDER_MIN <= y <= QRMatrix.FINDER_MAX
                )
                is_border = is_finder and (
                    x in QRMatrix.FINDER_BORDER or y in QRMatrix.FINDER_BORDER
                )
                is_center = is_finder and (
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
    def add_alignment(
        modules: list[list[Pixel]], kinds: list[list[Module | None]], row: int, col: int
    ) -> None:
        for y in range(-QRMatrix.ALIGNMENT_RADIUS, QRMatrix.ALIGNMENT_RADIUS + 1):
            for x in range(-QRMatrix.ALIGNMENT_RADIUS, QRMatrix.ALIGNMENT_RADIUS + 1):
                is_border = (
                    abs(x) == QRMatrix.ALIGNMENT_RADIUS
                    or abs(y) == QRMatrix.ALIGNMENT_RADIUS
                )
                is_center = x == 0 and y == 0
                QRMatrix.set_reserved(
                    modules, kinds, row + y, col + x, value=is_border or is_center
                )

    @staticmethod
    def add_data(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        bits: list[bool],
        *,
        mask: int = 0,
    ) -> None:
        bit_index = 0
        upward = True
        size = len(modules)
        for base_right_col in range(size - 1, 0, -2):
            right_col = (
                base_right_col - 1
                if base_right_col <= QRMatrix.TIMING_ROW_COL
                else base_right_col
            )
            row_order = range(size - 1, -1, -1) if upward else range(size)
            for row in row_order:
                for col in (right_col, right_col - 1):
                    if kinds[row][col] is not None:
                        continue
                    bit = bits[bit_index] if bit_index < len(bits) else False
                    modules[row][col] = bit ^ QRMatrix.mask(mask, row, col)
                    kinds[row][col] = "data"
                    bit_index += 1
            upward = not upward

    @staticmethod
    def add_format(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        version: QRVersion,
        *,
        mask: int = 0,
    ) -> None:
        size = version.size
        positions = [
            (0, 8),
            (1, 8),
            (2, 8),
            (3, 8),
            (4, 8),
            (5, 8),
            (7, 8),
            (8, 8),
            (8, 7),
            (8, 5),
            (8, 4),
            (8, 3),
            (8, 2),
            (8, 1),
            (8, 0),
        ]
        mirror_positions = [(8, size - 1 - index) for index in range(8)]
        mirror_positions.extend((size - 15 + index, 8) for index in range(8, 15))
        format_bits = QRMatrix.format_bits(version.error_correction, mask=mask)
        bits = [bool(format_bits & (1 << shift)) for shift in range(15)]
        for bit, (row, col), (mirror_row, mirror_col) in zip(
            bits, positions, mirror_positions, strict=True
        ):
            QRMatrix.set_reserved(modules, kinds, row, col, value=bit)
            QRMatrix.set_reserved(modules, kinds, mirror_row, mirror_col, value=bit)

    @staticmethod
    def reserve_format(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        version: QRVersion,
    ) -> None:
        size = version.size
        positions = [
            *((8, col) for col in range(9)),
            *((row, 8) for row in range(9)),
            *((size - 1 - index, 8) for index in range(7)),
            *((8, size - 8 + index) for index in range(8)),
        ]
        for row, col in positions:
            if kinds[row][col] is None:
                QRMatrix.set_reserved(modules, kinds, row, col, value=False)

    @staticmethod
    def reserve_version(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        version: QRVersion,
    ) -> None:
        bits = QRMatrix.version_bits(version.version)
        size = version.size
        for index in range(18):
            bit = bool(bits & (1 << index))
            row = index // 3
            col = (size - 11) + (index % 3)
            QRMatrix.set_reserved(modules, kinds, row, col, value=bit)
            QRMatrix.set_reserved(modules, kinds, col, row, value=bit)

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
    def in_bounds(row: int, col: int, size: int) -> bool:
        return 0 <= row < size and 0 <= col < size

    @staticmethod
    def mask(mask: int, row: int, col: int) -> bool:
        match mask:
            case 0:
                return (row + col) % 2 == 0
            case 1:
                return row % 2 == 0
            case 2:
                return col % 3 == 0
            case 3:
                return (row + col) % 3 == 0
            case 4:
                return ((row // 2) + (col // 3)) % 2 == 0
            case 5:
                return ((row * col) % 2) + ((row * col) % 3) == 0
            case 6:
                return (((row * col) % 2) + ((row * col) % 3)) % 2 == 0
            case 7:
                return (((row + col) % 2) + ((row * col) % 3)) % 2 == 0
            case _:
                msg = "mask must be between 0 and 7"
                raise ValueError(msg)

    @staticmethod
    def format_bits(error_correction: ErrorCorrection, *, mask: int) -> int:
        data = (QRVersions.FORMAT_ERROR_CORRECTION_BITS[error_correction] << 3) | mask
        remainder = data
        for _ in range(10):
            remainder = (remainder << 1) ^ ((remainder >> 9) * 0x537)
        return ((data << 10) | remainder) ^ 0x5412

    @staticmethod
    def version_bits(version: int) -> int:
        result = version << 12
        generator = 0x1F25
        for shift in range(17, 11, -1):
            if result & (1 << shift):
                result ^= generator << (shift - 12)
        return (version << 12) | result


@dataclass(frozen=True, kw_only=True)
class WifiPayload:
    """Build standard WiFi QR payload text.

    >>> print(WifiPayload.escape(r"semi;colon\\back:slash,comma"))
    semi\\;colon\\\\back\\:slash\\,comma
    >>> print(WifiPayload(ssid="Cafe;Net").text("secret"))
    WIFI:T:WPA;S:Cafe\\;Net;P:secret;;
    >>> WifiPayload(ssid="Cafe", auth="nopass").text("")
    'WIFI:T:nopass;S:Cafe;;'
    """

    ssid: str
    auth: WifiAuth = "WPA"
    hidden: bool = False

    def text(self, password: str) -> str:
        parts = [
            "WIFI:",
            f"T:{self.auth};",
            f"S:{self.escape(self.ssid)};",
        ]
        if self.auth != "nopass":
            parts.append(f"P:{self.escape(password)};")
        if self.hidden:
            parts.append("H:true;")
        parts.append(";")
        return "".join(parts)

    @staticmethod
    def escape(text: str) -> str:
        return "".join(
            f"\\{character}" if character in r"\;:," else character
            for character in text
        )


@dataclass(frozen=True, kw_only=True)
class QRJob:
    """Turn payload text into one or more QR codes.

    >>> byte_capacity = QRVersions.for_version(40, "L").capacity("byte")
    >>> [len(chunk.encode()) for chunk in QRJob().split_text("x" * (byte_capacity + 1), "byte")]
    [2953, 1]
    >>> version_1_capacity = QRVersions.for_version(1, "L").capacity("byte")
    >>> [len(chunk.encode()) for chunk in QRJob(version=1).split_text("x" * (version_1_capacity + 1), "byte")]
    [17, 1]
    >>> [code.version for code in QRJob(version=1).codes("x" * (version_1_capacity + 1))]
    [1, 1]
    >>> QRJob(version=1, split_mode="disabled").codes("x" * (version_1_capacity + 1))
    Traceback (most recent call last):
    ...
    ValueError: version 1-L byte QR codes support at most 17 UTF-8 bytes
    """

    error_correction: ErrorCorrection = "L"
    mode: RequestedMode = "auto"
    version: int | None = None
    split_mode: SplitMode = "all"

    def codes(self, text: str) -> list[QRCode]:
        segment = QRSegment.from_text(text, mode=self.mode)
        try:
            _ = self.version_metadata(segment)
        except ValueError:
            if self.split_mode == "disabled":
                raise
            return [
                QRCode.from_text(
                    chunk,
                    error_correction=self.error_correction,
                    mode=segment.mode,
                    version=self.version,
                )
                for chunk in self.split_text(text, segment.mode)
            ]
        return [
            QRCode.from_text(
                text,
                error_correction=self.error_correction,
                mode=self.mode,
                version=self.version,
            )
        ]

    def version_metadata(self, segment: QRSegment) -> QRVersion:
        if self.version is not None:
            metadata = QRVersions.for_version(self.version, self.error_correction)
            used_bits = segment.total_bits(metadata.version)
            if used_bits is not None and used_bits <= metadata.data_codewords * 8:
                return metadata
            msg = self.capacity_error(metadata, segment)
            raise ValueError(msg)
        return QRVersions.for_segment(segment, error_correction=self.error_correction)

    def capacity_error(self, metadata: QRVersion, segment: QRSegment) -> str:
        return (
            f"version {metadata.version}-{self.error_correction} {segment.mode} QR "
            f"codes support at most {metadata.capacity(segment.mode)} "
            f"{segment.unit_name}"
        )

    def split_text(
        self, text: str, mode: SegmentMode, *, capacity: int | None = None
    ) -> list[str]:
        max_capacity = capacity or QRVersions.for_version(
            self.version or QRVersions.MAX_VERSION, self.error_correction
        ).capacity(mode)
        if mode != "byte":
            return [
                text[index : index + max_capacity]
                for index in range(0, len(text), max_capacity)
            ] or [""]

        chunks: list[str] = []
        current: list[str] = []
        current_size = 0
        for character in text:
            character_size = len(character.encode())
            if current and current_size + character_size > max_capacity:
                chunks.append("".join(current))
                current = []
                current_size = 0
            if character_size > max_capacity:
                msg = "single character exceeds byte-mode QR capacity"
                raise ValueError(msg)
            current.append(character)
            current_size += character_size
        if current or not chunks:
            chunks.append("".join(current))
        return chunks


@dataclass(frozen=True, kw_only=True)
class OutputConfig:
    """Render a QR code using one configured output format.

    >>> OutputConfig(output_format="bits").render(QRCode.from_text("A")).splitlines()[0]
    '111111100101101111111'
    >>> OutputConfig(output_format="ascii", quiet_zone=0).render(QRCode.from_text("A")).splitlines()[0]
    '#######  # ## #######'
    """

    output_format: OutputFormat = "terminal"
    quiet_zone: int = 2
    terminal_image_protocol: TerminalImageProtocol = "auto"

    def render(self, code: QRCode) -> str | bytes:
        match self.output_format:
            case "terminal":
                return QRRenderer.render_terminal(
                    code.rows(), quiet_zone=self.quiet_zone
                )
            case "bits":
                return QRRenderer.render_bits(code.rows())
            case "ascii":
                return QRRenderer.render_ascii(code.rows(), quiet_zone=self.quiet_zone)
            case "svg":
                return QRRenderer.render_svg(code.rows(), quiet_zone=self.quiet_zone)
            case "bmp":
                return QRRenderer.render_bmp(code.rows(), quiet_zone=self.quiet_zone)
            case "png":
                return QRRenderer.render_png(code.rows(), quiet_zone=self.quiet_zone)
            case "terminal_img":
                return QRRenderer.render_terminal_image(
                    code.rows(),
                    quiet_zone=self.quiet_zone,
                    protocol=self.terminal_image_protocol,
                )
            case "html":
                return QRRenderer.render_html([code.rows()], quiet_zone=self.quiet_zone)


@dataclass(frozen=True, kw_only=True)
class OutputWriter:
    """Write rendered QR outputs to stdout or files.

    >>> writer = OutputWriter(output_format="svg", output="qr")
    >>> writer.output_path("svg")
    PosixPath('qr.svg')
    >>> writer = OutputWriter(output_format="png", output="qr")
    >>> writer.output_path("png", index=1, total=2)
    PosixPath('qr-2.png')
    """

    output_format: OutputFormat = "terminal"
    output: str | None = None
    split_mode: SplitMode = "all"
    quiet_zone: int = 2

    def write(
        self, rendered: list[str | bytes], rows_list: list[list[list[int]]]
    ) -> None:
        if self.output_format == "html":
            self.write_text_output(
                QRRenderer.render_html(rows_list, quiet_zone=self.quiet_zone),
                self.output_path("html"),
            )
            return

        if (
            self.output_format in {"terminal", "bits", "ascii", "terminal_img"}
            and self.output is None
        ):
            self.write_stdout_codes(cast(list[str], rendered))
            return

        ext = self.output_format
        for index, data in enumerate(rendered):
            path = self.output_path(ext, index=index, total=len(rendered))
            if isinstance(data, bytes):
                _ = path.write_bytes(data)
            else:
                self.write_text_output(data, path)

    @staticmethod
    def write_text_output(data: str, path: Path) -> None:
        _ = path.write_text(data, encoding="utf-8")

    def write_stdout_codes(self, rendered: list[str]) -> None:
        if self.split_mode != "wait" or len(rendered) <= 1:
            print("\n\n".join(rendered))
            return
        for index, data in enumerate(rendered):
            if index:
                self.wait_for_next_code()
            print(data)

    @staticmethod
    def wait_for_next_code() -> None:
        print("Press Enter for next QR code...", end="", file=sys.stderr, flush=True)
        try:
            with Path("/dev/tty").open(encoding="utf-8") as tty:
                _ = tty.readline()
        except OSError:
            with suppress(EOFError):
                _ = input()

    def output_path(
        self, extension: OutputFormat, *, index: int = 0, total: int = 1
    ) -> Path:
        base = Path(self.output or "output")
        suffix = f".{extension}"
        root = base.with_suffix("") if base.suffix == suffix else base
        if total > 1:
            root = Path(f"{root}-{index + 1}")
        return root if root.suffix == suffix else root.with_suffix(suffix)


@dataclass(frozen=True, kw_only=True)
class Args:
    """CLI adapter."""

    command: Command
    text: str | None = None
    wifi_ssid: str | None = None
    wifi_auth: WifiAuth | None = None
    wifi_hidden: bool = False
    quiet_zone: int
    error_correction: ErrorCorrection
    mode: RequestedMode | None = None
    output_format: OutputFormat
    terminal_image_protocol: TerminalImageProtocol
    version: int | None
    output: str | None
    split_mode: SplitMode

    def __post_init__(self) -> None:
        match self.command:
            case "wifi":
                assert self.wifi_ssid is not None
                assert self.wifi_auth is not None
                assert self.text is None
            case "text":
                assert self.wifi_ssid is None
                assert self.wifi_auth is None
            case _:
                assert_never(self.command)

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> Args:
        """Parse CLI arguments.

        >>> from contextlib import redirect_stderr
        >>> from io import StringIO
        >>> Args.from_argv(["--text", "HELLO", "--error-correction", "H"]).error_correction
        'H'
        >>> Args.from_argv(["--text", "123", "--mode", "numeric"]).mode
        'numeric'
        >>> Args.from_argv([]).mode
        'auto'
        >>> Args.from_argv(["--text", "A", "--format", "bits"]).output_format
        'bits'
        >>> Args.from_argv(["--text", "A", "--format", "png"]).output_format
        'png'
        >>> Args.from_argv(["--text", "A", "--format", "terminal_img"]).output_format
        'terminal_img'
        >>> Args.from_argv(["--text", "A", "--terminal-image-protocol", "iterm2"]).terminal_image_protocol
        'iterm2'
        >>> Args.from_argv(["--text", "A", "--version", "10"]).version
        10
        >>> Args.from_argv(["--text", "A", "--output", "qr.svg"]).output
        'qr.svg'
        >>> Args.from_argv(["--text", "A"]).wifi_ssid is None
        True
        >>> Args.from_argv(["wifi", "--ssid", "Cafe"]).command
        'wifi'
        >>> Args.from_argv(["wifi", "--ssid", "Cafe"]).text is None
        True
        >>> Args.from_argv([]).split_mode
        'all'
        >>> Args.from_argv(["--split-mode", "wait"]).split_mode
        'wait'
        >>> Args.from_argv(["--split-mode", "disabled"]).split_mode
        'disabled'
        >>> err = StringIO()
        >>> with redirect_stderr(err):
        ...     Args.from_argv(["--text", "A", "--version", "0"])
        Traceback (most recent call last):
        ...
        SystemExit: 2
        >>> err = StringIO()
        >>> with redirect_stderr(err):
        ...     Args.from_argv(["--text", "A", "--version", "41"])
        Traceback (most recent call last):
        ...
        SystemExit: 2
        >>> err = StringIO()
        >>> with redirect_stderr(err):
        ...     Args.from_argv(["A"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        SystemExit: 2
        >>> err = StringIO()
        >>> with redirect_stderr(err):
        ...     Args.from_argv(["--quiet-zone", "nope"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        SystemExit: 2
        """

        parser = argparse.ArgumentParser(description="Generate a terminal QR code.")
        cls.add_common_arguments(parser)
        namespace = parser.parse_args(argv)
        return cls(
            command=cast(Command, namespace.command),
            wifi_ssid=cast(str | None, namespace.ssid),
            wifi_auth=cast(WifiAuth | None, namespace.auth),
            wifi_hidden=cast(bool, namespace.hidden),
            quiet_zone=cast(int, namespace.quiet_zone),
            error_correction=cast(ErrorCorrection, namespace.error_correction),
            output_format=cast(OutputFormat, namespace.format),
            terminal_image_protocol=cast(
                TerminalImageProtocol, namespace.terminal_image_protocol
            ),
            version=cast(int | None, namespace.version),
            output=cast(str | None, namespace.output),
            split_mode=cast(SplitMode, namespace.split_mode),
            mode=cast(RequestedMode, namespace.mode),
            text=cast(str | None, namespace.text),
        )

    @classmethod
    def add_common_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.set_defaults(command="text", ssid=None, auth=None, hidden=False)
        _ = parser.add_argument("--text")
        _ = parser.add_argument(
            "-l", "--error-correction", choices=("L", "M", "Q", "H"), default="L"
        )
        _ = parser.add_argument("--version", type=cls.qr_version)
        _ = parser.add_argument(
            "--mode",
            choices=("auto", "numeric", "alphanumeric", "byte", "kanji"),
            default="auto",
        )
        _ = parser.add_argument("-q", "--quiet-zone", type=int, default=2)
        _ = parser.add_argument(
            "--format",
            choices=(
                "terminal",
                "bits",
                "ascii",
                "svg",
                "html",
                "bmp",
                "png",
                "terminal_img",
            ),
            default="terminal",
            help="output representation",
        )
        _ = parser.add_argument(
            "--terminal-image-protocol",
            choices=("auto", "kitty", "iterm2"),
            default="auto",
            help="terminal image protocol for --format terminal_img",
        )
        _ = parser.add_argument("--output", help="output file or basename")
        _ = parser.add_argument(
            "--split-mode",
            choices=("all", "wait", "disabled"),
            default="all",
            help="how to handle oversized input",
        )
        subparsers = parser.add_subparsers()
        wifi = subparsers.add_parser(
            "wifi", help="generate a QR code for WiFi credentials"
        )
        wifi.set_defaults(command="wifi")
        _ = wifi.add_argument("--ssid", required=True)
        _ = wifi.add_argument(
            "--auth",
            choices=("WPA", "WEP", "nopass"),
            default="WPA",
            help="WiFi authentication type",
        )
        _ = wifi.add_argument("--hidden", action="store_true")
        _ = wifi.add_argument(
            "-l", "--error-correction", choices=("L", "M", "Q", "H"), default="L"
        )
        _ = wifi.add_argument("--version", type=cls.qr_version)
        _ = wifi.add_argument("-q", "--quiet-zone", type=int, default=2)
        _ = wifi.add_argument(
            "--format",
            choices=(
                "terminal",
                "bits",
                "ascii",
                "svg",
                "html",
                "bmp",
                "png",
                "terminal_img",
            ),
            default="terminal",
            help="output representation",
        )
        _ = wifi.add_argument(
            "--terminal-image-protocol",
            choices=("auto", "kitty", "iterm2"),
            default="auto",
            help="terminal image protocol for --format terminal_img",
        )
        _ = wifi.add_argument("--output", help="output file or basename")
        _ = wifi.add_argument(
            "--split-mode",
            choices=("all", "wait", "disabled"),
            default="all",
            help="how to handle oversized input",
        )

    def main(self) -> int:
        """Generate and write QR code output.

        >>> from contextlib import redirect_stderr, redirect_stdout
        >>> from io import StringIO
        >>> from pathlib import Path
        >>> from unittest.mock import patch
        >>> import tempfile
        >>> out = StringIO()
        >>> with redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--text", "A", "--quiet-zone", "0"]).main()
        >>> exit_code
        0
        >>> out.getvalue().splitlines()[0].startswith(ANSI_BLACK * 7)
        True
        >>> out = StringIO()
        >>> with redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "bits"]).main()
        >>> exit_code
        0
        >>> out.getvalue().splitlines()[0]
        '111111100101101111111'
        >>> out = StringIO()
        >>> with redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "ascii", "--quiet-zone", "0"]).main()
        >>> exit_code
        0
        >>> out.getvalue().splitlines()[0]
        '#######  # ## #######'
        >>> out = StringIO()
        >>> with patch("sys.stdin", StringIO("A")), redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--quiet-zone", "3"]).main()
        >>> exit_code
        0
        >>> len(out.getvalue().splitlines())
        27
        >>> out = StringIO()
        >>> err = StringIO()
        >>> with patch("sys.stdin", StringIO("secret\\n")), redirect_stdout(out), redirect_stderr(err):
        ...     exit_code = Args.from_argv(["wifi", "--ssid", "Cafe", "--format", "bits"]).main()
        >>> exit_code
        0
        >>> err.getvalue()
        'Password: '
        >>> len(out.getvalue().splitlines()) > 21
        True
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "svg", "--output", f"{directory}/qr"]).main()
        ...     svg_path = Path(directory) / "qr.svg"
        ...     svg_exists = svg_path.exists() and svg_path.read_text().startswith("<svg ")
        >>> exit_code, svg_exists
        (0, True)
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "png", "--output", f"{directory}/qr"]).main()
        ...     png_path = Path(directory) / "qr.png"
        ...     png_header = png_path.read_bytes()[:8]
        >>> exit_code, png_header
        (0, b'\\x89PNG\\r\\n\\x1a\\n')
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "bmp", "--output", f"{directory}/qr"]).main()
        ...     bmp_path = Path(directory) / "qr.bmp"
        ...     bmp_header = bmp_path.read_bytes()[:2]
        >>> exit_code, bmp_header
        (0, b'BM')
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "html", "--output", f"{directory}/qr"]).main()
        ...     html_path = Path(directory) / "qr.html"
        ...     html_has_svg = "<svg " in html_path.read_text()
        >>> exit_code, html_has_svg
        (0, True)
        >>> out = StringIO()
        >>> with redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "terminal_img", "--terminal-image-protocol", "kitty"]).main()
        >>> exit_code, out.getvalue().startswith("\\x1b_Ga=T,f=100;")
        (0, True)
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     exit_code = Args.from_argv(["--text", "A", "--format", "terminal_img", "--terminal-image-protocol", "kitty", "--output", f"{directory}/qr"]).main()
        ...     stream_path = Path(directory) / "qr.terminal_img"
        ...     stream_starts = stream_path.read_text().startswith("\\x1b_Ga=T,f=100;")
        >>> exit_code, stream_starts
        (0, True)
        >>> out = StringIO()
        >>> err = StringIO()
        >>> long_text = "x" * (QRVersions.for_version(40, "L").capacity("byte") + 1)
        >>> with redirect_stdout(out), redirect_stderr(err):
        ...     exit_code = Args.from_argv(["--text", long_text, "--format", "terminal_img", "--terminal-image-protocol", "kitty"]).main()
        >>> terminal_img_output = out.getvalue()
        >>> exit_code, terminal_img_output.count("\\x1b_Ga=T,f=100") == 2, err.getvalue()
        (0, True, '')
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     long_text = "x" * (QRVersions.for_version(40, "L").capacity("byte") + 1)
        ...     exit_code = Args.from_argv(["--text", long_text, "--format", "terminal_img", "--terminal-image-protocol", "kitty", "--output", f"{directory}/long"]).main()
        ...     paths = sorted(path.name for path in Path(directory).iterdir())
        ...     first_is_chunked = (Path(directory) / "long-1.terminal_img").read_text().startswith("\\x1b_Ga=T,f=100,m=1;")
        >>> exit_code, paths, first_is_chunked
        (0, ['long-1.terminal_img', 'long-2.terminal_img'], True)
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     long_text = "x" * (QRVersions.for_version(40, "L").capacity("byte") + 1)
        ...     exit_code = Args.from_argv(["--text", long_text, "--format", "svg", "--output", f"{directory}/long"]).main()
        ...     paths = sorted(path.name for path in Path(directory).iterdir())
        >>> exit_code, paths
        (0, ['long-1.svg', 'long-2.svg'])
        >>> out = StringIO()
        >>> long_text = "x" * (QRVersions.for_version(40, "L").capacity("byte") + 1)
        >>> with redirect_stdout(out):
        ...     exit_code = Args.from_argv(["--text", long_text, "--format", "bits"]).main()
        >>> exit_code
        0
        >>> "" in out.getvalue().splitlines()
        True
        >>> out = StringIO()
        >>> err = StringIO()
        >>> with patch("pathlib.Path.open", side_effect=OSError), patch("sys.stdin", StringIO("")), redirect_stdout(out), redirect_stderr(err):
        ...     exit_code = Args.from_argv(["--text", long_text, "--format", "bits", "--split-mode", "wait"]).main()
        >>> exit_code
        0
        >>> err.getvalue()
        'Press Enter for next QR code...'
        >>> "" not in out.getvalue().splitlines()
        True
        """

        codes = self.codes()
        renderer = self.output_config()
        rendered = [renderer.render(code) for code in codes]
        self.writer().write(rendered, [code.rows() for code in codes])
        return 0

    def codes(self) -> list[QRCode]:
        return self.job().codes(self.payload_text())

    def payload_text(self) -> str:
        """Return the text payload from CLI text, stdin, or WiFi fields.

        >>> from contextlib import redirect_stderr
        >>> from io import StringIO
        >>> from unittest.mock import patch
        >>> Args.from_argv(["--text", "A"]).payload_text()
        'A'
        >>> with patch("sys.stdin", StringIO("from stdin")):
        ...     Args.from_argv([]).payload_text()
        'from stdin'
        >>> err = StringIO()
        >>> with patch("sys.stdin", StringIO("secret\\n")), redirect_stderr(err):
        ...     Args.from_argv(["wifi", "--ssid", "Cafe"]).payload_text()
        'WIFI:T:WPA;S:Cafe;P:secret;;'
        >>> err.getvalue()
        'Password: '
        """

        match self.command:
            case "text":
                return self.text if self.text is not None else sys.stdin.read()
            case "wifi":
                if self.wifi_ssid is None or self.wifi_auth is None:
                    msg = "wifi ssid is required"
                    raise ValueError(msg)
                password = (
                    "" if self.wifi_auth == "nopass" else self.read_wifi_password()
                )
                return WifiPayload(
                    ssid=self.wifi_ssid, auth=self.wifi_auth, hidden=self.wifi_hidden
                ).text(password)
            case _:
                assert_never(self.command)

    def job(self) -> QRJob:
        """Build the QR generation job for the selected command.

        >>> Args.from_argv(["--text", "A", "--version", "10"]).job().codes("A")[0].version
        10
        >>> Args.from_argv(["wifi", "--ssid", "Cafe"]).job().mode
        'byte'
        """

        if self.command == "wifi":
            return QRJob(
                error_correction=self.error_correction,
                mode="byte",
                version=self.version,
                split_mode=self.split_mode,
            )
        else:
            assert self.mode is not None
            return QRJob(
                error_correction=self.error_correction,
                mode=self.mode,
                version=self.version,
                split_mode=self.split_mode,
            )

    def output_config(self) -> OutputConfig:
        return OutputConfig(
            output_format=self.output_format,
            quiet_zone=self.quiet_zone,
            terminal_image_protocol=self.terminal_image_protocol,
        )

    def writer(self) -> OutputWriter:
        return OutputWriter(
            output_format=self.output_format,
            output=self.output,
            split_mode=self.split_mode,
            quiet_zone=self.quiet_zone,
        )

    @staticmethod
    def read_wifi_password() -> str:
        """Read WiFi passwords without echoing interactive input.

        >>> from contextlib import redirect_stderr
        >>> from io import StringIO
        >>> from unittest.mock import patch
        >>> with patch("sys.stdin.isatty", return_value=True), patch("getpass.getpass", return_value="hidden") as getpass_mock:
        ...     Args.read_wifi_password()
        'hidden'
        >>> getpass_mock.assert_called_once_with("Password: ")
        >>> err = StringIO()
        >>> with patch("sys.stdin.isatty", return_value=False), patch("sys.stdin", StringIO("secret\\n")), redirect_stderr(err):
        ...     Args.read_wifi_password()
        'secret'
        >>> err.getvalue()
        'Password: '
        """

        if sys.stdin.isatty():
            return getpass.getpass("Password: ")
        else:
            print("Password: ", end="", file=sys.stderr, flush=True)
            return sys.stdin.read().rstrip("\n")

    @staticmethod
    def qr_version(value: str) -> int:
        try:
            version = int(value)
        except ValueError as error:
            msg = "version must be between 1 and 40"
            raise argparse.ArgumentTypeError(msg) from error
        if not QRVersions.MIN_VERSION <= version <= QRVersions.MAX_VERSION:
            msg = "version must be between 1 and 40"
            raise argparse.ArgumentTypeError(msg)
        return version


if __name__ == "__main__":
    raise SystemExit(Args.from_argv().main())
