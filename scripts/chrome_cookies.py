"""Extract Chrome cookies for a given domain on macOS.

Uses the Keychain to decrypt Chrome's cookie database.
Returns cookies as a dict suitable for requests.Session.
"""

import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes


def _get_chrome_key() -> bytes:
    """Get Chrome Safe Storage key from macOS Keychain."""
    raw = subprocess.check_output(
        ["security", "find-generic-password", "-w", "-s", "Chrome Safe Storage", "-a", "Chrome"],
        stderr=subprocess.DEVNULL,
    )
    return raw.strip()


def _derive_aes_key(chrome_key: bytes) -> bytes:
    """Derive AES-128-CBC key from Chrome's storage key."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA1(),
        length=16,
        salt=b"saltysalt",
        iterations=1003,
    )
    return kdf.derive(chrome_key)


def _decrypt_cookie(encrypted_value: bytes, aes_key: bytes) -> str:
    """Decrypt a Chrome cookie value (v10 format on macOS).

    Chrome prepends a 32-byte application-bound prefix to the plaintext
    before AES-128-CBC encryption. We skip those bytes after decryption.
    """
    if not encrypted_value:
        return ""
    # v10 prefix (3 bytes) on macOS
    if encrypted_value[:3] == b"v10":
        encrypted_value = encrypted_value[3:]
    else:
        # Unencrypted or unknown format
        return encrypted_value.decode("utf-8", errors="replace")

    iv = b" " * 16  # 16 space bytes
    cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(encrypted_value) + decryptor.finalize()

    # Remove PKCS7 padding
    pad_len = decrypted[-1]
    if isinstance(pad_len, int) and 1 <= pad_len <= 16:
        decrypted = decrypted[:-pad_len]

    # Skip 32-byte app-bound encryption prefix (Chrome 127+)
    if len(decrypted) > 32:
        decrypted = decrypted[32:]

    return decrypted.decode("utf-8", errors="replace")


def get_cookies(domain: str) -> dict[str, str]:
    """Extract all cookies for a domain from Chrome.

    Returns dict of {name: value} suitable for requests.Session.cookies.
    """
    chrome_key = _get_chrome_key()
    aes_key = _derive_aes_key(chrome_key)

    # Chrome locks the cookie DB — copy to temp file
    cookie_db = Path.home() / "Library/Application Support/Google/Chrome/Default/Cookies"
    if not cookie_db.exists():
        # Try Profile 1, etc.
        for profile in ["Profile 1", "Profile 2", "Profile 3"]:
            alt = Path.home() / f"Library/Application Support/Google/Chrome/{profile}/Cookies"
            if alt.exists():
                cookie_db = alt
                break

    if not cookie_db.exists():
        raise FileNotFoundError(f"Chrome cookie database not found at {cookie_db}")

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copy2(cookie_db, tmp_path)

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.execute(
            "SELECT name, encrypted_value FROM cookies WHERE host_key LIKE ?",
            (f"%{domain}%",),
        )
        cookies = {}
        for name, encrypted_value in cursor.fetchall():
            cookies[name] = _decrypt_cookie(encrypted_value, aes_key)
        conn.close()
        return cookies
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    cookies = get_cookies("kanfocus.com")
    print(f"Found {len(cookies)} cookies for kanfocus.com:")
    for name in sorted(cookies):
        val = cookies[name]
        print(f"  {name} = {val[:20]}{'...' if len(val) > 20 else ''}")
