from __future__ import annotations

import ctypes
import hashlib
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BLOCK_SIZE = 16
BUFFER_SIZE = 4096
HEADER_SIZE = 16
DIGEST_SIZE = 32

ARIA = 0x10
AES = 0x20
AES_ECB = 0x30
KEY128 = 0x01
KEY192 = 0x02
KEY256 = 0x03
ARIA_128_CBC = ARIA | KEY128
ARIA_192_CBC = ARIA | KEY192
ARIA_256_CBC = ARIA | KEY256
AES_128_CBC = AES | KEY128
AES_192_CBC = AES | KEY192
AES_256_CBC = AES | KEY256
AES_128_ECB = AES_ECB | KEY128
AES_192_ECB = AES_ECB | KEY192
AES_256_ECB = AES_ECB | KEY256

ENCRYPT_TYPES = {
    ARIA_128_CBC,
    ARIA_192_CBC,
    ARIA_256_CBC,
    AES_128_CBC,
    AES_192_CBC,
    AES_256_CBC,
    AES_128_ECB,
    AES_192_ECB,
    AES_256_ECB,
}
COMPRESS_TYPES = {0x00, 0x01}

XCNKEY = b"7TQF1N4UKXOKDNNG"
XCNKEY_FIELD = b"XCNKEY="
KEYFILE_SIZE = 80
NATIVE_ARIA_LIBRARY = Path(__file__).resolve().parent / "native" / "libxcnaria.so"


class XcnCryptoError(Exception):
    pass


class NativeAria:
    def __init__(self, library_path: Path):
        self.library = ctypes.CDLL(str(library_path))
        self.library.xcn_aria_decrypt_blocks.argtypes = (
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )
        self.library.xcn_aria_decrypt_blocks.restype = ctypes.c_int
        self.library.xcn_aria_encrypt_blocks.argtypes = (
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
        )
        self.library.xcn_aria_encrypt_blocks.restype = ctypes.c_int

    def decrypt_blocks(self, key: bytes, key_bits: int, data: bytes) -> bytes:
        if len(data) % BLOCK_SIZE != 0:
            raise XcnCryptoError("ARIA input length must be a multiple of 16 bytes")
        output = ctypes.create_string_buffer(len(data))
        result = self.library.xcn_aria_decrypt_blocks(key, key_bits, data, len(data), output)
        if result != 0:
            raise XcnCryptoError(f"native ARIA decrypt failed: {result}")
        return output.raw

    def encrypt_blocks(self, key: bytes, key_bits: int, data: bytes) -> bytes:
        if len(data) % BLOCK_SIZE != 0:
            raise XcnCryptoError("ARIA input length must be a multiple of 16 bytes")
        output = ctypes.create_string_buffer(len(data))
        result = self.library.xcn_aria_encrypt_blocks(key, key_bits, data, len(data), output)
        if result != 0:
            raise XcnCryptoError(f"native ARIA encrypt failed: {result}")
        return output.raw


@lru_cache(maxsize=1)
def _native_aria() -> NativeAria | None:
    mode = os.getenv("XCN_CRYPTO_NATIVE", "auto").strip().lower()
    if mode in {"0", "false", "no", "off", "disabled"}:
        return None
    try:
        return NativeAria(NATIVE_ARIA_LIBRARY)
    except OSError as exc:
        if mode in {"1", "true", "yes", "on", "required"}:
            raise XcnCryptoError(f"native ARIA library is not available: {NATIVE_ARIA_LIBRARY}") from exc
        return None


@dataclass(frozen=True)
class XcnHeader:
    version: int
    encrypt_type: int
    compress_type: int
    content_length: int

    @property
    def pad_size(self) -> int:
        mod = self.content_length % BLOCK_SIZE
        return 0 if mod == 0 else BLOCK_SIZE - mod

    @property
    def encrypted_length(self) -> int:
        return int(math.ceil(self.content_length / BLOCK_SIZE) * BLOCK_SIZE)


def is_encrypted_bytes(data: bytes) -> bool:
    try:
        parse_header(data[:HEADER_SIZE])
        return True
    except XcnCryptoError:
        return False


def is_encrypted_file(path: Path) -> bool:
    with path.open("rb") as file:
        return is_encrypted_bytes(file.read(HEADER_SIZE))


def parse_header(data: bytes) -> XcnHeader:
    if len(data) < HEADER_SIZE:
        raise XcnCryptoError("XCN header is too short")
    if data[:3] != b"XCN" or data[3] != 0x01:
        raise XcnCryptoError("not an XCN encrypted file")
    version = data[4]
    encrypt_type = data[5]
    compress_type = data[6]
    content_length = int.from_bytes(data[8:16], "big")
    if encrypt_type not in ENCRYPT_TYPES:
        raise XcnCryptoError(f"unsupported XCN encrypt type: 0x{encrypt_type:02x}")
    if compress_type not in COMPRESS_TYPES:
        raise XcnCryptoError(f"unsupported XCN compress type: 0x{compress_type:02x}")
    if compress_type != 0:
        raise XcnCryptoError("compressed XCN files are not supported yet")
    return XcnHeader(version, encrypt_type, compress_type, content_length)


def load_key_file(path: str | Path) -> bytes:
    key_path = Path(path)
    data = key_path.read_bytes()
    if len(data) != KEYFILE_SIZE:
        raise XcnCryptoError(f"invalid XCN key file size: {key_path}")

    out = _decrypt_aria_blocks(XCNKEY, 128, data[:48])

    if bytes(out[: len(XCNKEY_FIELD)]) != XCNKEY_FIELD:
        raise XcnCryptoError("invalid XCN key file marker")

    digest_source = bytes(out[: len(XCNKEY_FIELD) + DIGEST_SIZE])
    expected = hashlib.sha256(digest_source).digest()
    actual = data[KEYFILE_SIZE - DIGEST_SIZE :]
    if actual != expected:
        raise XcnCryptoError("invalid XCN key file digest")
    return bytes(out[len(XCNKEY_FIELD) : len(XCNKEY_FIELD) + DIGEST_SIZE])


def default_key_path() -> Path:
    configured = os.getenv("XCN_CRYPTO_KEY_FILE", "").strip()
    if configured:
        return Path(configured)
    return Path("/models/enckey")


def decrypt_file(input_path: str | Path, output_path: str | Path, key: bytes | None = None) -> bool:
    source = Path(input_path)
    target = Path(output_path)
    data = source.read_bytes()
    if not is_encrypted_bytes(data):
        return False
    decrypted = decrypt_bytes(data, key=key)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(decrypted)
    return True


def decrypt_bytes(data: bytes, key: bytes | None = None) -> bytes:
    header = parse_header(data[:HEADER_SIZE])
    if header.encrypt_type != ARIA_256_CBC:
        raise XcnCryptoError(f"unsupported XCN file cipher: 0x{header.encrypt_type:02x}")
    if key is None:
        key = load_key_file(default_key_path())
    if len(key) < 32:
        raise XcnCryptoError("XCN ARIA-256 key must be at least 32 bytes")

    encrypted_start = HEADER_SIZE
    encrypted_end = encrypted_start + header.encrypted_length
    digest_end = encrypted_end + DIGEST_SIZE
    if len(data) < digest_end:
        raise XcnCryptoError("XCN file is truncated")

    encrypted = data[encrypted_start:encrypted_end]
    plaintext = _decrypt_aria_blocks(key, 256, encrypted)[: header.content_length]

    expected_digest = data[encrypted_end:digest_end]
    actual_digest = hashlib.sha256(plaintext).digest()
    if expected_digest != actual_digest:
        raise XcnCryptoError("XCN file digest mismatch")
    return plaintext


def _decrypt_aria_blocks(key: bytes, key_bits: int, data: bytes) -> bytes:
    native = _native_aria()
    if native is not None:
        return native.decrypt_blocks(key[: key_bits // 8], key_bits, data)

    engine = AriaEngine(key_bits)
    engine.set_key(key)
    output = bytearray()
    for offset in range(0, len(data), BLOCK_SIZE):
        output.extend(engine.decrypt_block(data[offset : offset + BLOCK_SIZE]))
    return bytes(output)


def _encrypt_aria_blocks(key: bytes, key_bits: int, data: bytes) -> bytes:
    native = _native_aria()
    if native is not None:
        return native.encrypt_blocks(key[: key_bits // 8], key_bits, data)

    engine = AriaEngine(key_bits)
    engine.set_key(key)
    output = bytearray()
    for offset in range(0, len(data), BLOCK_SIZE):
        output.extend(engine.encrypt_block(data[offset : offset + BLOCK_SIZE]))
    return bytes(output)


class AriaEngine:
    KRK = (
        (0x517CC1B7, 0x27220A94, 0xFE13ABE8, 0xFA9A6EE0),
        (0x6DB14ACC, 0x9E21C820, 0xFF28B1D5, 0xEF5DE2B0),
        (0xDB92371D, 0x2126E970, 0x03249775, 0x04E8C90E),
    )

    S1: list[int] = []
    S2: list[int] = []
    X1: list[int] = []
    X2: list[int] = []
    TS1: list[int] = []
    TS2: list[int] = []
    TX1: list[int] = []
    TX2: list[int] = []

    def __init__(self, key_size: int):
        self.key_size = key_size
        if key_size == 128:
            self.number_of_rounds = 12
        elif key_size == 192:
            self.number_of_rounds = 14
        elif key_size == 256:
            self.number_of_rounds = 16
        else:
            raise XcnCryptoError(f"invalid ARIA key size: {key_size}")
        self.master_key = b""
        self.enc_round_keys: list[int] | None = None
        self.dec_round_keys: list[int] | None = None
        self._ensure_tables()

    @classmethod
    def _ensure_tables(cls) -> None:
        if cls.S1:
            return
        exp = [0] * 256
        log = [0] * 256
        exp[0] = 1
        for i in range(1, 256):
            value = (exp[i - 1] << 1) ^ exp[i - 1]
            if value & 0x100:
                value ^= 0x11B
            exp[i] = value
        for i in range(1, 255):
            log[exp[i]] = i

        a_matrix = (
            (1, 0, 0, 0, 1, 1, 1, 1),
            (1, 1, 0, 0, 0, 1, 1, 1),
            (1, 1, 1, 0, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 1),
            (1, 1, 1, 1, 1, 0, 0, 0),
            (0, 1, 1, 1, 1, 1, 0, 0),
            (0, 0, 1, 1, 1, 1, 1, 0),
            (0, 0, 0, 1, 1, 1, 1, 1),
        )
        b_matrix = (
            (0, 1, 0, 1, 1, 1, 1, 0),
            (0, 0, 1, 1, 1, 1, 0, 1),
            (1, 1, 0, 1, 0, 1, 1, 1),
            (1, 0, 0, 1, 1, 1, 0, 1),
            (0, 0, 1, 0, 1, 1, 0, 0),
            (1, 0, 0, 0, 0, 0, 0, 1),
            (0, 1, 0, 1, 1, 1, 0, 1),
            (1, 1, 0, 1, 0, 0, 1, 1),
        )

        cls.S1 = [0] * 256
        cls.S2 = [0] * 256
        cls.X1 = [0] * 256
        cls.X2 = [0] * 256
        for i in range(256):
            p = 0 if i == 0 else exp[255 - log[i]]
            value = 0
            for j in range(8):
                bit = 0
                for k in range(8):
                    if (p >> (7 - k)) & 0x01:
                        bit ^= a_matrix[k][j]
                value = (value << 1) ^ bit
            value ^= 0x63
            cls.S1[i] = value & 0xFF
            cls.X1[value & 0xFF] = i

        for i in range(256):
            p = 0 if i == 0 else exp[(247 * log[i]) % 255]
            value = 0
            for j in range(8):
                bit = 0
                for k in range(8):
                    if (p >> k) & 0x01:
                        bit ^= b_matrix[7 - j][k]
                value = (value << 1) ^ bit
            value ^= 0xE2
            cls.S2[i] = value & 0xFF
            cls.X2[value & 0xFF] = i

        cls.TS1 = [cls._u32(0x00010101 * cls.S1[i]) for i in range(256)]
        cls.TS2 = [cls._u32(0x01000101 * cls.S2[i]) for i in range(256)]
        cls.TX1 = [cls._u32(0x01010001 * cls.X1[i]) for i in range(256)]
        cls.TX2 = [cls._u32(0x01010100 * cls.X2[i]) for i in range(256)]

    @staticmethod
    def _u32(value: int) -> int:
        return value & 0xFFFFFFFF

    @classmethod
    def _to_int(cls, data: bytes, offset: int = 0) -> int:
        return cls._u32((data[offset] << 24) ^ (data[offset + 1] << 16) ^ (data[offset + 2] << 8) ^ data[offset + 3])

    @classmethod
    def _to_bytes(cls, value: int) -> bytes:
        value &= 0xFFFFFFFF
        return bytes(((value >> 24) & 0xFF, (value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF))

    @classmethod
    def _m(cls, value: int) -> int:
        return cls._u32(
            0x00010101 * ((value >> 24) & 0xFF)
            ^ 0x01000101 * ((value >> 16) & 0xFF)
            ^ 0x01010001 * ((value >> 8) & 0xFF)
            ^ 0x01010100 * (value & 0xFF)
        )

    @classmethod
    def _badc(cls, value: int) -> int:
        return cls._u32(((value << 8) & 0xFF00FF00) ^ ((value >> 8) & 0x00FF00FF))

    @classmethod
    def _cdab(cls, value: int) -> int:
        return cls._u32(((value << 16) & 0xFFFF0000) ^ ((value >> 16) & 0x0000FFFF))

    @classmethod
    def _dcba(cls, value: int) -> int:
        return cls._u32(
            ((value & 0x000000FF) << 24)
            ^ ((value & 0x0000FF00) << 8)
            ^ ((value & 0x00FF0000) >> 8)
            ^ ((value & 0xFF000000) >> 24)
        )

    @classmethod
    def _diff_block(cls, values: list[int], offset: int) -> list[int]:
        t0 = cls._m(values[offset])
        t1 = cls._m(values[offset + 1])
        t2 = cls._m(values[offset + 2])
        t3 = cls._m(values[offset + 3])
        t1 ^= t2
        t2 ^= t3
        t0 ^= t1
        t3 ^= t1
        t2 ^= t0
        t1 ^= t2
        t1 = cls._badc(t1)
        t2 = cls._cdab(t2)
        t3 = cls._dcba(t3)
        t1 ^= t2
        t2 ^= t3
        t0 ^= t1
        t3 ^= t1
        t2 ^= t0
        t1 ^= t2
        return [cls._u32(t0), cls._u32(t1), cls._u32(t2), cls._u32(t3)]

    @classmethod
    def _swap_and_diffuse(cls, values: list[int], offset1: int, offset2: int) -> None:
        first = cls._diff_block(values, offset1)
        second = cls._diff_block(values, offset2)
        values[offset1 : offset1 + 4] = second
        values[offset2 : offset2 + 4] = first

    @classmethod
    def _gsrk(cls, x_values: list[int], y_values: list[int], rot: int, rk: list[int], offset: int) -> None:
        q = 4 - (rot // 32)
        r = rot % 32
        s = 32 - r
        rk[offset] = cls._u32(x_values[0] ^ (y_values[q % 4] >> r) ^ (y_values[(q + 3) % 4] << s))
        rk[offset + 1] = cls._u32(x_values[1] ^ (y_values[(q + 1) % 4] >> r) ^ (y_values[q % 4] << s))
        rk[offset + 2] = cls._u32(x_values[2] ^ (y_values[(q + 2) % 4] >> r) ^ (y_values[(q + 1) % 4] << s))
        rk[offset + 3] = cls._u32(x_values[3] ^ (y_values[(q + 3) % 4] >> r) ^ (y_values[(q + 2) % 4] << s))

    def set_key(self, master_key: bytes) -> None:
        if len(master_key) * 8 < self.key_size:
            raise XcnCryptoError("ARIA master key is too short")
        self.master_key = bytes(master_key)
        self.enc_round_keys = None
        self.dec_round_keys = None
        self._setup_round_keys()

    def _setup_round_keys(self) -> None:
        self.enc_round_keys = [0] * (4 * (self.number_of_rounds + 1))
        self._do_enc_key_setup(self.enc_round_keys)
        self.dec_round_keys = list(self.enc_round_keys)
        self._do_dec_key_setup(self.dec_round_keys)

    def _do_enc_key_setup(self, rk: list[int]) -> None:
        mk = self.master_key
        w0 = [self._to_int(mk, 0), self._to_int(mk, 4), self._to_int(mk, 8), self._to_int(mk, 12)]
        q = (self.key_size - 128) // 64
        t = [self._u32(w0[i] ^ self.KRK[q][i]) for i in range(4)]
        t = self._fo(t)

        if self.key_size > 128:
            w1 = [self._to_int(mk, 16), self._to_int(mk, 20), 0, 0]
            if self.key_size > 192:
                w1[2] = self._to_int(mk, 24)
                w1[3] = self._to_int(mk, 28)
        else:
            w1 = [0, 0, 0, 0]
        w1 = [self._u32(w1[i] ^ t[i]) for i in range(4)]

        q = 0 if q == 2 else q + 1
        t = [self._u32(w1[i] ^ self.KRK[q][i]) for i in range(4)]
        t = self._fe(t)
        w2 = [self._u32(t[i] ^ w0[i]) for i in range(4)]

        q = 0 if q == 2 else q + 1
        t = [self._u32(w2[i] ^ self.KRK[q][i]) for i in range(4)]
        t = self._fo(t)
        w3 = [self._u32(t[i] ^ w1[i]) for i in range(4)]

        j = 0
        for x, y, rot in (
            (w0, w1, 19),
            (w1, w2, 19),
            (w2, w3, 19),
            (w3, w0, 19),
            (w0, w1, 31),
            (w1, w2, 31),
            (w2, w3, 31),
            (w3, w0, 31),
            (w0, w1, 67),
            (w1, w2, 67),
            (w2, w3, 67),
            (w3, w0, 67),
            (w0, w1, 97),
        ):
            self._gsrk(x, y, rot, rk, j)
            j += 4
        if self.key_size > 128:
            self._gsrk(w1, w2, 97, rk, j)
            j += 4
            self._gsrk(w2, w3, 97, rk, j)
            j += 4
        if self.key_size > 192:
            self._gsrk(w3, w0, 97, rk, j)
            j += 4
            self._gsrk(w0, w1, 109, rk, j)

    def _do_dec_key_setup(self, rk: list[int]) -> None:
        a = 0
        z = 32 + self.key_size // 8
        rk[a : a + 4], rk[z : z + 4] = rk[z : z + 4], rk[a : a + 4]
        a += 4
        z -= 4
        while a < z:
            self._swap_and_diffuse(rk, a, z)
            a += 4
            z -= 4
        rk[a : a + 4] = self._diff_block(rk, a)

    def _fo(self, t: list[int]) -> list[int]:
        t0, t1, t2, t3 = t
        t0 = self.TS1[(t0 >> 24) & 0xFF] ^ self.TS2[(t0 >> 16) & 0xFF] ^ self.TX1[(t0 >> 8) & 0xFF] ^ self.TX2[t0 & 0xFF]
        t1 = self.TS1[(t1 >> 24) & 0xFF] ^ self.TS2[(t1 >> 16) & 0xFF] ^ self.TX1[(t1 >> 8) & 0xFF] ^ self.TX2[t1 & 0xFF]
        t2 = self.TS1[(t2 >> 24) & 0xFF] ^ self.TS2[(t2 >> 16) & 0xFF] ^ self.TX1[(t2 >> 8) & 0xFF] ^ self.TX2[t2 & 0xFF]
        t3 = self.TS1[(t3 >> 24) & 0xFF] ^ self.TS2[(t3 >> 16) & 0xFF] ^ self.TX1[(t3 >> 8) & 0xFF] ^ self.TX2[t3 & 0xFF]
        return self._mix(t0, t1, t2, t3, first=True)

    def _fe(self, t: list[int]) -> list[int]:
        t0, t1, t2, t3 = t
        t0 = self.TX1[(t0 >> 24) & 0xFF] ^ self.TX2[(t0 >> 16) & 0xFF] ^ self.TS1[(t0 >> 8) & 0xFF] ^ self.TS2[t0 & 0xFF]
        t1 = self.TX1[(t1 >> 24) & 0xFF] ^ self.TX2[(t1 >> 16) & 0xFF] ^ self.TS1[(t1 >> 8) & 0xFF] ^ self.TS2[t1 & 0xFF]
        t2 = self.TX1[(t2 >> 24) & 0xFF] ^ self.TX2[(t2 >> 16) & 0xFF] ^ self.TS1[(t2 >> 8) & 0xFF] ^ self.TS2[t2 & 0xFF]
        t3 = self.TX1[(t3 >> 24) & 0xFF] ^ self.TX2[(t3 >> 16) & 0xFF] ^ self.TS1[(t3 >> 8) & 0xFF] ^ self.TS2[t3 & 0xFF]
        return self._mix(t0, t1, t2, t3, first=False)

    def _mix(self, t0: int, t1: int, t2: int, t3: int, *, first: bool) -> list[int]:
        t1 ^= t2
        t2 ^= t3
        t0 ^= t1
        t3 ^= t1
        t2 ^= t0
        t1 ^= t2
        if first:
            t1 = self._badc(t1)
            t2 = self._cdab(t2)
            t3 = self._dcba(t3)
        else:
            t3 = self._badc(t3)
            t0 = self._cdab(t0)
            t1 = self._dcba(t1)
        t1 ^= t2
        t2 ^= t3
        t0 ^= t1
        t3 ^= t1
        t2 ^= t0
        t1 ^= t2
        return [self._u32(t0), self._u32(t1), self._u32(t2), self._u32(t3)]

    def decrypt_block(self, block: bytes) -> bytes:
        if len(block) != BLOCK_SIZE:
            raise XcnCryptoError("ARIA block must be exactly 16 bytes")
        if self.dec_round_keys is None:
            raise XcnCryptoError("ARIA key is not initialized")
        return self._crypt_block(block, self.dec_round_keys)

    def encrypt_block(self, block: bytes) -> bytes:
        if len(block) != BLOCK_SIZE:
            raise XcnCryptoError("ARIA block must be exactly 16 bytes")
        if self.enc_round_keys is None:
            raise XcnCryptoError("ARIA key is not initialized")
        return self._crypt_block(block, self.enc_round_keys)

    def _crypt_block(self, block: bytes, rk: list[int]) -> bytes:
        t0 = self._to_int(block, 0)
        t1 = self._to_int(block, 4)
        t2 = self._to_int(block, 8)
        t3 = self._to_int(block, 12)
        j = 0
        for _ in range(1, self.number_of_rounds // 2):
            t0 ^= rk[j]
            t1 ^= rk[j + 1]
            t2 ^= rk[j + 2]
            t3 ^= rk[j + 3]
            j += 4
            t0, t1, t2, t3 = self._fo([t0, t1, t2, t3])

            t0 ^= rk[j]
            t1 ^= rk[j + 1]
            t2 ^= rk[j + 2]
            t3 ^= rk[j + 3]
            j += 4
            t0, t1, t2, t3 = self._fe([t0, t1, t2, t3])

        t0 ^= rk[j]
        t1 ^= rk[j + 1]
        t2 ^= rk[j + 2]
        t3 ^= rk[j + 3]
        j += 4
        t0, t1, t2, t3 = self._fo([t0, t1, t2, t3])

        t0 ^= rk[j]
        t1 ^= rk[j + 1]
        t2 ^= rk[j + 2]
        t3 ^= rk[j + 3]
        j += 4

        return bytes(
            (
                self.X1[(t0 >> 24) & 0xFF] ^ ((rk[j] >> 24) & 0xFF),
                self.X2[(t0 >> 16) & 0xFF] ^ ((rk[j] >> 16) & 0xFF),
                self.S1[(t0 >> 8) & 0xFF] ^ ((rk[j] >> 8) & 0xFF),
                self.S2[t0 & 0xFF] ^ (rk[j] & 0xFF),
                self.X1[(t1 >> 24) & 0xFF] ^ ((rk[j + 1] >> 24) & 0xFF),
                self.X2[(t1 >> 16) & 0xFF] ^ ((rk[j + 1] >> 16) & 0xFF),
                self.S1[(t1 >> 8) & 0xFF] ^ ((rk[j + 1] >> 8) & 0xFF),
                self.S2[t1 & 0xFF] ^ (rk[j + 1] & 0xFF),
                self.X1[(t2 >> 24) & 0xFF] ^ ((rk[j + 2] >> 24) & 0xFF),
                self.X2[(t2 >> 16) & 0xFF] ^ ((rk[j + 2] >> 16) & 0xFF),
                self.S1[(t2 >> 8) & 0xFF] ^ ((rk[j + 2] >> 8) & 0xFF),
                self.S2[t2 & 0xFF] ^ (rk[j + 2] & 0xFF),
                self.X1[(t3 >> 24) & 0xFF] ^ ((rk[j + 3] >> 24) & 0xFF),
                self.X2[(t3 >> 16) & 0xFF] ^ ((rk[j + 3] >> 16) & 0xFF),
                self.S1[(t3 >> 8) & 0xFF] ^ ((rk[j + 3] >> 8) & 0xFF),
                self.S2[t3 & 0xFF] ^ (rk[j + 3] & 0xFF),
            )
        )
