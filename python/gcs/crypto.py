"""AES-256-GCM authenticated encryption + mTLS context for GCS commands (§8).

Gap 7 fix: the paper specifies AES-256-GCM for waypoint command encryption
and TLS 1.3 mutual authentication for the GCS ↔ drone link.

Key management:
  - 256-bit symmetric key derived from a pre-shared secret via HKDF-SHA256.
  - 96-bit random nonce per message (NIST SP 800-38D recommendation).
  - AAD = drone_id (4 bytes, big-endian) to bind ciphertext to a specific drone.
  - Wire format: nonce (12 B) || tag (16 B) || ciphertext.

mTLS:
  - `build_server_ssl_context` loads the GCS cert/key + CA bundle.
  - `build_client_ssl_context` loads the drone cert/key + CA bundle.
  - Both enforce TLS 1.3 minimum and require client certificates.
"""
from __future__ import annotations

import os
import ssl
import struct
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# ── Constants ─────────────────────────────────────────────────────────────────

_NONCE_LEN = 12   # 96-bit nonce (NIST SP 800-38D)
_TAG_LEN   = 16   # 128-bit authentication tag
_KEY_LEN   = 32   # 256-bit AES key


# ── Key derivation ────────────────────────────────────────────────────────────

def derive_key(psk: bytes, salt: bytes | None = None, info: bytes = b"cellhawk-aes256gcm") -> bytes:
    """Derive a 256-bit AES key from a pre-shared secret via HKDF-SHA256.

    Args:
        psk:  Pre-shared key material (≥ 16 bytes recommended).
        salt: Optional random salt (32 bytes recommended). None → HKDF default.
        info: Context label binding the key to this application.

    Returns:
        32-byte AES-256 key.
    """
    hkdf = HKDF(algorithm=SHA256(), length=_KEY_LEN, salt=salt, info=info)
    return hkdf.derive(psk)


# ── Encryption / Decryption ───────────────────────────────────────────────────

@dataclass(frozen=True)
class EncryptedFrame:
    """Wire-format encrypted command frame.

    Layout: nonce (12 B) || tag (16 B) || ciphertext (variable).
    The tag is embedded by AESGCM inside the ciphertext output; we expose it
    separately for clarity but store it contiguously on the wire.
    """
    nonce:      bytes   # 12 bytes
    ciphertext: bytes   # includes 16-byte GCM tag appended by AESGCM

    def to_bytes(self) -> bytes:
        """Serialise to wire format: nonce || ciphertext (tag embedded)."""
        return self.nonce + self.ciphertext

    @classmethod
    def from_bytes(cls, data: bytes) -> "EncryptedFrame":
        """Deserialise from wire format."""
        if len(data) < _NONCE_LEN + _TAG_LEN:
            raise ValueError(f"Frame too short: {len(data)} bytes")
        return cls(nonce=data[:_NONCE_LEN], ciphertext=data[_NONCE_LEN:])


class CommandCipher:
    """AES-256-GCM cipher for GCS ↔ drone command frames.

    Thread-safe: each encrypt call generates a fresh random nonce.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != _KEY_LEN:
            raise ValueError(f"Key must be {_KEY_LEN} bytes, got {len(key)}")
        self._aes = AESGCM(key)

    # ── public API ────────────────────────────────────────────────────────────

    def encrypt(self, plaintext: bytes, drone_id: int) -> EncryptedFrame:
        """Encrypt a command payload bound to a specific drone.

        Args:
            plaintext: Raw command bytes (e.g. JSON-encoded waypoint).
            drone_id:  Drone identifier used as Additional Authenticated Data.

        Returns:
            EncryptedFrame ready for transmission.
        """
        nonce = os.urandom(_NONCE_LEN)
        aad   = _drone_aad(drone_id)
        ct    = self._aes.encrypt(nonce, plaintext, aad)
        return EncryptedFrame(nonce=nonce, ciphertext=ct)

    def decrypt(self, frame: EncryptedFrame, drone_id: int) -> bytes:
        """Decrypt and authenticate a command frame.

        Args:
            frame:    EncryptedFrame received from the wire.
            drone_id: Must match the drone_id used during encryption.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            cryptography.exceptions.InvalidTag: if authentication fails.
        """
        aad = _drone_aad(drone_id)
        return self._aes.decrypt(frame.nonce, frame.ciphertext, aad)

    def encrypt_bytes(self, plaintext: bytes, drone_id: int) -> bytes:
        """Convenience: encrypt and return raw wire bytes."""
        return self.encrypt(plaintext, drone_id).to_bytes()

    def decrypt_bytes(self, data: bytes, drone_id: int) -> bytes:
        """Convenience: decrypt from raw wire bytes."""
        return self.decrypt(EncryptedFrame.from_bytes(data), drone_id)


# ── mTLS context builders ─────────────────────────────────────────────────────

def build_server_ssl_context(
    certfile: str | Path,
    keyfile:  str | Path,
    cafile:   str | Path,
) -> ssl.SSLContext:
    """Build a TLS 1.3 server context that requires client certificates.

    Args:
        certfile: Path to the GCS server certificate (PEM).
        keyfile:  Path to the GCS server private key (PEM).
        cafile:   Path to the CA bundle used to verify drone client certs (PEM).

    Returns:
        Configured ssl.SSLContext for use with uvicorn / websockets.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    ctx.load_verify_locations(cafile=str(cafile))
    ctx.verify_mode = ssl.CERT_REQUIRED
    return ctx


def build_client_ssl_context(
    certfile: str | Path,
    keyfile:  str | Path,
    cafile:   str | Path,
) -> ssl.SSLContext:
    """Build a TLS 1.3 client context presenting a drone certificate.

    Args:
        certfile: Path to the drone client certificate (PEM).
        keyfile:  Path to the drone client private key (PEM).
        cafile:   Path to the CA bundle used to verify the GCS server cert (PEM).

    Returns:
        Configured ssl.SSLContext for use with websockets / httpx.
    """
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    ctx.load_cert_chain(certfile=str(certfile), keyfile=str(keyfile))
    ctx.load_verify_locations(cafile=str(cafile))
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    return ctx


# ── Helpers ───────────────────────────────────────────────────────────────────

def _drone_aad(drone_id: int) -> bytes:
    """4-byte big-endian drone_id used as Additional Authenticated Data."""
    return struct.pack(">I", drone_id & 0xFFFF_FFFF)
