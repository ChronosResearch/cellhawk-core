"""Tests for Gap 7: mTLS + AES-256-GCM command encryption (python/gcs/crypto.py)."""
from __future__ import annotations

import os
import struct

import pytest
from cryptography.exceptions import InvalidTag

from python.gcs.crypto import (
    CommandCipher,
    EncryptedFrame,
    _KEY_LEN,
    _NONCE_LEN,
    _TAG_LEN,
    _drone_aad,
    derive_key,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def key() -> bytes:
    return os.urandom(_KEY_LEN)


@pytest.fixture()
def cipher(key: bytes) -> CommandCipher:
    return CommandCipher(key)


DRONE_ID = 42
PLAINTEXT = b'{"type":"waypoint","east_m":100.0,"north_m":200.0}'


# ── derive_key ────────────────────────────────────────────────────────────────

def test_derive_key_length() -> None:
    key = derive_key(b"super-secret-psk")
    assert len(key) == _KEY_LEN


def test_derive_key_deterministic_with_same_salt() -> None:
    salt = os.urandom(32)
    k1 = derive_key(b"psk", salt=salt)
    k2 = derive_key(b"psk", salt=salt)
    assert k1 == k2


def test_derive_key_different_salts_differ() -> None:
    k1 = derive_key(b"psk", salt=os.urandom(32))
    k2 = derive_key(b"psk", salt=os.urandom(32))
    assert k1 != k2


def test_derive_key_different_info_differs() -> None:
    k1 = derive_key(b"psk", info=b"app-a")
    k2 = derive_key(b"psk", info=b"app-b")
    assert k1 != k2


# ── CommandCipher construction ────────────────────────────────────────────────

def test_cipher_rejects_wrong_key_length() -> None:
    with pytest.raises(ValueError, match="Key must be"):
        CommandCipher(b"short")


def test_cipher_accepts_32_byte_key(key: bytes) -> None:
    CommandCipher(key)  # must not raise


# ── Encrypt / Decrypt round-trip ──────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    recovered = cipher.decrypt(frame, DRONE_ID)
    assert recovered == PLAINTEXT


def test_encrypt_bytes_decrypt_bytes_roundtrip(cipher: CommandCipher) -> None:
    wire = cipher.encrypt_bytes(PLAINTEXT, DRONE_ID)
    recovered = cipher.decrypt_bytes(wire, DRONE_ID)
    assert recovered == PLAINTEXT


def test_each_encrypt_produces_unique_nonce(cipher: CommandCipher) -> None:
    f1 = cipher.encrypt(PLAINTEXT, DRONE_ID)
    f2 = cipher.encrypt(PLAINTEXT, DRONE_ID)
    assert f1.nonce != f2.nonce


def test_ciphertext_differs_per_nonce(cipher: CommandCipher) -> None:
    f1 = cipher.encrypt(PLAINTEXT, DRONE_ID)
    f2 = cipher.encrypt(PLAINTEXT, DRONE_ID)
    assert f1.ciphertext != f2.ciphertext


# ── Authentication failures ───────────────────────────────────────────────────

def test_wrong_drone_id_fails_authentication(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    with pytest.raises(InvalidTag):
        cipher.decrypt(frame, DRONE_ID + 1)


def test_tampered_ciphertext_fails_authentication(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    # Flip a byte in the ciphertext
    tampered = bytearray(frame.ciphertext)
    tampered[0] ^= 0xFF
    bad_frame = EncryptedFrame(nonce=frame.nonce, ciphertext=bytes(tampered))
    with pytest.raises(InvalidTag):
        cipher.decrypt(bad_frame, DRONE_ID)


def test_tampered_nonce_fails_authentication(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    bad_nonce = bytes(b ^ 0xFF for b in frame.nonce)
    bad_frame = EncryptedFrame(nonce=bad_nonce, ciphertext=frame.ciphertext)
    with pytest.raises(InvalidTag):
        cipher.decrypt(bad_frame, DRONE_ID)


def test_wrong_key_fails_authentication(key: bytes) -> None:
    cipher_a = CommandCipher(key)
    cipher_b = CommandCipher(os.urandom(_KEY_LEN))
    frame = cipher_a.encrypt(PLAINTEXT, DRONE_ID)
    with pytest.raises(InvalidTag):
        cipher_b.decrypt(frame, DRONE_ID)


# ── Wire format ───────────────────────────────────────────────────────────────

def test_wire_format_length(cipher: CommandCipher) -> None:
    wire = cipher.encrypt_bytes(PLAINTEXT, DRONE_ID)
    # nonce + tag + plaintext
    assert len(wire) == _NONCE_LEN + _TAG_LEN + len(PLAINTEXT)


def test_wire_format_nonce_is_first_12_bytes(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    wire = frame.to_bytes()
    assert wire[:_NONCE_LEN] == frame.nonce


def test_encrypted_frame_from_bytes_roundtrip(cipher: CommandCipher) -> None:
    frame = cipher.encrypt(PLAINTEXT, DRONE_ID)
    wire = frame.to_bytes()
    restored = EncryptedFrame.from_bytes(wire)
    assert restored.nonce == frame.nonce
    assert restored.ciphertext == frame.ciphertext


def test_encrypted_frame_from_bytes_too_short() -> None:
    with pytest.raises(ValueError, match="Frame too short"):
        EncryptedFrame.from_bytes(b"\x00" * (_NONCE_LEN + _TAG_LEN - 1))


# ── AAD helper ────────────────────────────────────────────────────────────────

def test_drone_aad_is_4_bytes() -> None:
    assert len(_drone_aad(1)) == 4


def test_drone_aad_big_endian() -> None:
    assert _drone_aad(1) == struct.pack(">I", 1)
    assert _drone_aad(0xDEAD_BEEF) == struct.pack(">I", 0xDEAD_BEEF)


def test_drone_aad_wraps_at_32_bits() -> None:
    # drone_id masked to 32 bits
    assert _drone_aad(0x1_0000_0001) == _drone_aad(1)


# ── TelemetryHub integration ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_telemetry_hub_encrypts_outbound(key: bytes) -> None:
    """TelemetryHub with cipher must encrypt send_command payloads."""
    from unittest.mock import AsyncMock, MagicMock

    from python.gcs.telemetry import TelemetryHub

    cipher = CommandCipher(key)
    hub = TelemetryHub(cipher=cipher)

    # Inject a mock WebSocket
    mock_ws = MagicMock()
    mock_ws.send_bytes = AsyncMock()
    hub._connections[DRONE_ID] = mock_ws

    await hub.send_command(DRONE_ID, PLAINTEXT)

    sent: bytes = mock_ws.send_bytes.call_args[0][0]
    # Must not be plaintext
    assert sent != PLAINTEXT
    # Must be decryptable
    recovered = cipher.decrypt_bytes(sent, DRONE_ID)
    assert recovered == PLAINTEXT


@pytest.mark.asyncio
async def test_telemetry_hub_no_cipher_sends_plaintext() -> None:
    """TelemetryHub without cipher sends raw bytes unchanged."""
    from unittest.mock import AsyncMock, MagicMock

    from python.gcs.telemetry import TelemetryHub

    hub = TelemetryHub()
    mock_ws = MagicMock()
    mock_ws.send_bytes = AsyncMock()
    hub._connections[DRONE_ID] = mock_ws

    await hub.send_command(DRONE_ID, PLAINTEXT)
    sent: bytes = mock_ws.send_bytes.call_args[0][0]
    assert sent == PLAINTEXT
