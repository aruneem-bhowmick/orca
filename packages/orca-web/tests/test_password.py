"""Tests for orca_web.auth.password."""

from orca_web.auth.password import hash_password, verify_password


class TestHashPassword:
    def test_returns_bcrypt_hash(self):
        h = hash_password("secret123")
        assert h.startswith("$2b$")

    def test_different_salts_each_call(self):
        h1 = hash_password("same")
        h2 = hash_password("same")
        assert h1 != h2


class TestVerifyPassword:
    def test_correct_password_returns_true(self):
        h = hash_password("correct-horse")
        assert verify_password("correct-horse", h) is True

    def test_wrong_password_returns_false(self):
        h = hash_password("correct-horse")
        assert verify_password("wrong-horse", h) is False

    def test_empty_password_hashes_and_verifies(self):
        h = hash_password("")
        assert verify_password("", h) is True
        assert verify_password("x", h) is False
