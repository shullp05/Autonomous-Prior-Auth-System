
import sys
from unittest.mock import MagicMock

# Mock missing dependencies
sys.modules["langchain_core"] = MagicMock()
sys.modules["langchain_core.documents"] = MagicMock()
sys.modules["langchain_ollama"] = MagicMock()
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["BCEmbedding"] = MagicMock()

import pytest
import threading
from agent_logic import _BCE_LOCK, _BCE_MODEL, _get_bce_model
from unittest.mock import patch, MagicMock

def test_bce_lock_exists():
    assert isinstance(_BCE_LOCK, type(threading.Lock()))

def test_bce_model_lazy_loading_with_lock():
    # If _BCE_MODEL is None, accessing it should use the lock
    # We can't easily verify the lock was acquired without inspecting internals, 
    # but we can verify it doesn't crash and returns the mock.
    
    with patch("BCEmbedding.RerankerModel") as mock_cls:
        # Reset global if needed (unsafe in parallel tests, but okay for sequential)
        # agent_logic._BCE_MODEL = None 
        # Actually importing it might have cached the None value in our namespace, 
        # but check the module usage.
        
        # We'll rely on the fact that if RerankerModel init is called, our mock intercepts it.
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        
        # Force None (be careful if running in parallel)
        import agent_logic
        agent_logic._BCE_MODEL = None
        
        model = agent_logic._get_bce_model()
        assert model == mock_instance
        assert agent_logic._BCE_MODEL == mock_instance
        
        # Second call should return same instance without init
        model2 = agent_logic._get_bce_model()
        assert model2 == model
        assert mock_cls.call_count == 1 # Only called once

def test_safe_loinc_constants():
    from config import LOINC_BMI, LOINC_HEIGHT, LOINC_WEIGHT
    assert LOINC_BMI == "39156-5"
    assert LOINC_WEIGHT == "29463-7"
    assert LOINC_HEIGHT == "8302-2"
