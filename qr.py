#!/usr/bin/env python3
################################################################
# Copyright (c) 2026 Witalis Domitrz <witekdomitrz@gmail.com>
# AGPL License
################################################################

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from itertools import cycle
from typing import ClassVar, Literal, cast

Module = Literal["data", "reserved"]
Pixel = bool | None
ErrorCorrection = Literal["L", "M", "Q", "H"]
SegmentMode = Literal["numeric", "alphanumeric", "byte", "kanji"]
RequestedMode = Literal["auto", "numeric", "alphanumeric", "byte", "kanji"]


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
    [7, 14, 8]
    >>> ReedSolomon.remainder([32, 91, 11, 120, 209, 114, 220], 7)
    [209, 239, 196, 207, 78, 195, 109]
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
                if remainder >= self.NUMERIC_REMAINDER_ONE_DIGIT:
                    return (groups * 3) + 1
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
    ERROR_INDEX: ClassVar[dict[ErrorCorrection, int]] = {"L": 0, "M": 1, "Q": 2, "H": 3}
    FORMAT_BITS: ClassVar[dict[ErrorCorrection, int]] = {"L": 1, "M": 0, "Q": 3, "H": 2}
    ECC_CODEWORDS_PER_BLOCK: ClassVar[tuple[tuple[int, ...], ...]] = (
        (
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
        (
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
        (
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
        (
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
    )
    ERROR_BLOCKS: ClassVar[tuple[tuple[int, ...], ...]] = (
        (
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
        (
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
        (
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
        (
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
    )
    ALIGNMENT_POSITIONS: ClassVar[tuple[tuple[int, ...], ...]] = (
        (),
        (6, 18),
        (6, 22),
        (6, 26),
        (6, 30),
        (6, 34),
        (6, 22, 38),
        (6, 24, 42),
        (6, 26, 46),
        (6, 28, 50),
        (6, 30, 54),
        (6, 32, 58),
        (6, 34, 62),
        (6, 26, 46, 66),
        (6, 26, 48, 70),
        (6, 26, 50, 74),
        (6, 30, 54, 78),
        (6, 30, 56, 82),
        (6, 30, 58, 86),
        (6, 34, 62, 90),
        (6, 28, 50, 72, 94),
        (6, 26, 50, 74, 98),
        (6, 30, 54, 78, 102),
        (6, 28, 54, 80, 106),
        (6, 32, 58, 84, 110),
        (6, 30, 58, 86, 114),
        (6, 34, 62, 90, 118),
        (6, 26, 50, 74, 98, 122),
        (6, 30, 54, 78, 102, 126),
        (6, 26, 52, 78, 104, 130),
        (6, 30, 56, 82, 108, 134),
        (6, 34, 60, 86, 112, 138),
        (6, 30, 58, 86, 114, 142),
        (6, 34, 62, 90, 118, 146),
        (6, 30, 54, 78, 102, 126, 150),
        (6, 24, 50, 76, 102, 128, 154),
        (6, 28, 54, 80, 106, 132, 158),
        (6, 32, 58, 84, 110, 136, 162),
        (6, 26, 54, 82, 110, 138, 166),
        (6, 30, 58, 86, 114, 142, 170),
    )

    @classmethod
    def for_version(
        cls, version: int, error_correction: ErrorCorrection = "L"
    ) -> QRVersion:
        if not cls.MIN_VERSION <= version <= cls.MAX_VERSION:
            msg = "QR version must be between 1 and 40"
            raise ValueError(msg)
        index = cls.ERROR_INDEX[error_correction]
        raw_codewords = cls.raw_data_modules(version) // 8
        error_codewords = cls.ECC_CODEWORDS_PER_BLOCK[index][version]
        block_count = cls.ERROR_BLOCKS[index][version]
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
            alignment_positions=list(cls.ALIGNMENT_POSITIONS[version - 1]),
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
    COUNT_BITS: ClassVar[dict[SegmentMode, tuple[int, int, int]]] = {
        "numeric": (10, 12, 14),
        "alphanumeric": (9, 11, 13),
        "byte": (8, 16, 16),
        "kanji": (8, 10, 12),
    }
    SHIFT_JIS_BYTES_PER_KANJI: ClassVar[int] = 2
    KANJI_RANGE_1_MIN: ClassVar[int] = 0x8140
    KANJI_RANGE_1_MAX: ClassVar[int] = 0x9FFC
    KANJI_RANGE_1_OFFSET: ClassVar[int] = 0x8140
    KANJI_RANGE_2_MIN: ClassVar[int] = 0xE040
    KANJI_RANGE_2_MAX: ClassVar[int] = 0xEBBF
    KANJI_RANGE_2_OFFSET: ClassVar[int] = 0xC140

    @property
    def unit_name(self) -> str:
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
        if cls.ALPHANUMERIC_RE.fullmatch(text) is not None:
            return "alphanumeric"
        if cls.can_encode_kanji(text):
            return "kanji"
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
        return cls(mode="numeric", character_count=len(text), bits=buffer.bits or [])

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
        return cls(
            mode="alphanumeric", character_count=len(text), bits=buffer.bits or []
        )

    @classmethod
    def byte(cls, text: str) -> QRSegment:
        buffer = BitBuffer()
        data = text.encode()
        for byte in data:
            buffer.append(byte, 8)
        return cls(mode="byte", character_count=len(data), bits=buffer.bits or [])

    @classmethod
    def kanji(cls, text: str) -> QRSegment:
        buffer = BitBuffer()
        for character in text:
            value = cls.kanji_value(character)
            if value is None:
                msg = "text cannot be encoded in kanji mode"
                raise ValueError(msg)
            buffer.append(value, 13)
        return cls(mode="kanji", character_count=len(text), bits=buffer.bits or [])

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
        return cls.COUNT_BITS[mode][(version + 7) // 17]

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
    >>> rendered = code.render(quiet_zone=1)
    >>> rendered.count("\\n")
    22
    >>> QRCode.from_text("1" * 42).version
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
    >>> QRCode.from_text("1234567890").mode
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
    """

    text: str
    version: int
    error_correction: ErrorCorrection
    mode: SegmentMode
    data_codewords: list[int]
    error_codewords: list[int]
    matrix: list[list[bool]]

    DEFAULT_ERROR_CORRECTION: ClassVar[ErrorCorrection] = "L"
    DEFAULT_MODE: ClassVar[RequestedMode] = "byte"

    @property
    def size(self) -> int:
        return len(self.matrix)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        error_correction: ErrorCorrection = DEFAULT_ERROR_CORRECTION,
        mode: RequestedMode = DEFAULT_MODE,
    ) -> QRCode:
        segment = QRSegment.from_text(text, mode=mode)
        version = QRVersions.for_segment(segment, error_correction=error_correction)
        data_codewords = cls.encode_segment(segment, version)
        blocks = cls.make_blocks(data_codewords, version)
        codewords = cls.interleave_blocks(blocks)
        error_codewords = [codeword for _, error in blocks for codeword in error]
        matrix = QRMatrix.from_codewords(codewords, version)
        return cls(
            text=text,
            version=version.version,
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
        segment = QRSegment(
            mode="byte", character_count=len(data), bits=buffer.bits or []
        )
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

        remaining = (version.data_codewords * 8) - len(buffer.bits or [])
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

    def render(
        self,
        *,
        quiet_zone: int = 4,
        black: str = "  ",
        white: str = "██",
    ) -> str:
        """Render the QR matrix as terminal text.

        >>> small = QRCode.from_text("A").render(quiet_zone=0, black="1", white="0")
        >>> small.splitlines()[0]
        '111111100101101111111'
        >>> len(small.splitlines())
        21
        >>> QRCode.from_text("A").render(quiet_zone=1).splitlines()[0] == "██" * 23
        True
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
                if kinds[row][col] is None:
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
        modules: list[list[Pixel]], kinds: list[list[Module | None]], bits: list[bool]
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
                    modules[row][col] = bit ^ QRMatrix.mask(row, col)
                    kinds[row][col] = "data"
                    bit_index += 1
            upward = not upward

    @staticmethod
    def add_format(
        modules: list[list[Pixel]],
        kinds: list[list[Module | None]],
        version: QRVersion,
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
        format_bits = QRMatrix.format_bits(version.error_correction, mask=0)
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
    def mask(row: int, col: int) -> bool:
        return (row + col) % 2 == 0

    @staticmethod
    def format_bits(error_correction: ErrorCorrection, *, mask: int) -> int:
        data = (QRVersions.FORMAT_BITS[error_correction] << 3) | mask
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
class Args:
    """CLI adapter.

    >>> from contextlib import redirect_stderr, redirect_stdout
    >>> from io import StringIO
    >>> from unittest.mock import patch
    >>> out = StringIO()
    >>> with redirect_stdout(out):
    ...     exit_code = Args.from_argv(["--text", "A", "--quiet-zone", "0"]).main()
    >>> exit_code
    0
    >>> out.getvalue().splitlines()[0]
    '              ████  ██    ██              '
    >>> Args.from_argv(["--text", "HELLO", "--error-correction", "H"]).error_correction
    'H'
    >>> Args.from_argv(["--text", "123", "--mode", "numeric"]).mode
    'numeric'
    >>> out = StringIO()
    >>> with patch("sys.stdin", StringIO("from stdin")), redirect_stdout(out):
    ...     exit_code = Args.from_argv(["--quiet-zone", "0"]).main()
    >>> exit_code
    0
    >>> len(out.getvalue().splitlines())
    21
    >>> out = StringIO()
    >>> with patch("sys.stdin", StringIO("A")), redirect_stdout(out):
    ...     exit_code = Args.from_argv(["--quiet-zone", "3"]).main()
    >>> exit_code
    0
    >>> len(out.getvalue().splitlines())
    27
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

    text: str | None = None
    quiet_zone: int = 4
    error_correction: ErrorCorrection = "L"
    mode: RequestedMode = "byte"

    @classmethod
    def from_argv(cls, argv: list[str] | None = None) -> Args:
        parser = argparse.ArgumentParser(description="Generate a terminal QR code.")
        parser.add_argument("--text")
        parser.add_argument(
            "--error-correction", choices=("L", "M", "Q", "H"), default="L"
        )
        parser.add_argument(
            "--mode",
            choices=("auto", "numeric", "alphanumeric", "byte", "kanji"),
            default="byte",
        )
        parser.add_argument("--quiet-zone", type=int, default=4)
        namespace = parser.parse_args(argv)
        return cls(
            text=namespace.text,
            quiet_zone=namespace.quiet_zone,
            error_correction=cast("ErrorCorrection", namespace.error_correction),
            mode=cast("RequestedMode", namespace.mode),
        )

    def main(self) -> int:
        text = self.text if self.text is not None else sys.stdin.read()
        print(
            QRCode.from_text(
                text, error_correction=self.error_correction, mode=self.mode
            ).render(quiet_zone=self.quiet_zone)
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(Args.from_argv().main())
