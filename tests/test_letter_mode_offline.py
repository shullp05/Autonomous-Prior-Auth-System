import socket

import pytest

from priorauth.letter_service import LetterResult, generate_approved_letter
from priorauth.offline_mode import OfflineModeError, disable_offline, enforce_offline


def _sample_payload():
    patient_data = {
        "patient_id": "TEST_PATIENT",
        "name": "Test Patient",
        "dob": "1970-01-01",
    }
    findings = {
        "bmi_numeric": 32.0,
        "comorbidity_category": "NONE",
    }
    return patient_data, findings


def _skip_if_socket_denied() -> None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        pytest.skip("Socket creation blocked in this environment; cannot validate offline sockets.")
    else:
        s.close()


@pytest.fixture(autouse=True)
def _reset_offline(monkeypatch: pytest.MonkeyPatch):
    disable_offline()
    yield
    disable_offline()
    monkeypatch.delenv("PA_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("OFFLINE_MODE", raising=False)
    monkeypatch.delenv("PA_OFFLINE_ALLOW_LOCALHOST", raising=False)
    monkeypatch.delenv("PA_OFFLINE_STRICT_UNKNOWN_HOST", raising=False)
    monkeypatch.delenv("PA_LETTER_MODE", raising=False)
    monkeypatch.delenv("PA_ALLOW_LETTER_FALLBACK", raising=False)


def test_deterministic_letter_mode_blocks_all_sockets(monkeypatch):
    _skip_if_socket_denied()
    monkeypatch.setenv("PA_LETTER_MODE", "deterministic")
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "false")
    monkeypatch.setenv("PA_OFFLINE_STRICT_UNKNOWN_HOST", "true")

    enforce_offline(force=True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(OfflineModeError):
            s.connect(("127.0.0.1", 11434))

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(OfflineModeError):
            s.connect(("8.8.8.8", 53))

    patient_data, findings = _sample_payload()
    res = generate_approved_letter(patient_data, "Meets criteria.", findings)

    assert isinstance(res, LetterResult)
    assert res.status == "DETERMINISTIC"
    assert res.letter is not None


def test_ollama_letter_mode_allows_loopback_only(monkeypatch):
    _skip_if_socket_denied()
    monkeypatch.setenv("PA_LETTER_MODE", "ollama")
    monkeypatch.setenv("PA_ALLOW_LETTER_FALLBACK", "false")
    monkeypatch.setenv("PA_OFFLINE_MODE", "true")
    monkeypatch.setenv("PA_OFFLINE_ALLOW_LOCALHOST", "true")
    monkeypatch.setenv("PA_OFFLINE_STRICT_UNKNOWN_HOST", "true")

    enforce_offline(force=True)

    # External host should be blocked
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(OfflineModeError):
            s.connect(("8.8.8.8", 53))

    # Loopback should be allowed (may still be connection-refused if service isn't running)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", 11434))
        except OfflineModeError as e:
            pytest.fail(f"Loopback should be allowed but was blocked: {e}")
        except OSError:
            pass

    patient_data, findings = _sample_payload()
    res = generate_approved_letter(patient_data, "Meets criteria.", findings)

    assert isinstance(res, LetterResult)
    assert res.status in {"OLLAMA", "LLM_UNAVAILABLE"}
    assert res.status != "FALLBACK_USED"
    if res.status == "OLLAMA":
        assert res.letter is not None
    else:
        assert res.letter is None
