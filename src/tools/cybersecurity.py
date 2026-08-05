import hashlib
import hmac

def encrypt_rsa(message: str, public_key: tuple) -> list:
    """RSA encryption using modular exponentiation"""
    n, e = public_key
    return [(ord(char) ** e) % n for char in message]

def generate_sha256(text: str) -> str:
    """Generates a SHA-256 hash"""
    return hashlib.sha256(text.encode()).hexdigest()

def verify_hmac(key: bytes, message: bytes, signature: str) -> bool:
    """Verifies HMAC signature for network security"""
    expected = hmac.new(key, message, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
