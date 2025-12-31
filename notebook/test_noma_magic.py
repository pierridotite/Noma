"""
Tests for NOMA Jupyter Magic extension.

Run with:
    python test_noma_magic.py
    
or with pytest:
    python -m pytest test_noma_magic.py
"""

import tempfile
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Add the noma_magic package to path
sys.path.insert(0, str(Path(__file__).parent))

from noma_magic.magic import NomaExecutor


class TestNomaExecutor:
    """Test the NomaExecutor class."""
    
    def test_init(self):
        """Test executor initialization."""
        executor = NomaExecutor()
        assert executor.work_dir.exists()
        assert executor.cache_dir.exists()
        assert executor.artifacts_dir.exists()
    
    def test_hash_content(self):
        """Test content hashing."""
        executor = NomaExecutor()
        hash1 = executor.hash_content("code1")
        hash2 = executor.hash_content("code1")
        hash3 = executor.hash_content("code2")
        
        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 16  # Should be 16 chars (SHA256[:16])
    
    def test_cache_path(self):
        """Test cache path generation."""
        executor = NomaExecutor()
        cache_path = executor.get_cache_path("abc123", "interpreter")
        
        assert "abc123" in str(cache_path)
        assert "interpreter" in str(cache_path)
    
    def test_find_noma_binary(self):
        """Test finding NOMA binary."""
        executor = NomaExecutor()
        
        try:
            binary = executor.find_noma_binary()
            assert binary is not None
            # Should be a string path
            assert isinstance(binary, str)
        except FileNotFoundError:
            # Expected if NOMA isn't built
            print("[SKIP] test_find_noma_binary (NOMA binary not found)")
            return
    
    def test_save_execution_log(self):
        """Test execution log writing."""
        executor = NomaExecutor()
        executor.save_execution_log(
            cell_num=1,
            code="test code",
            output="test output",
            error=None,
            mode="interpreter",
            success=True
        )
        
        assert executor.log_file.exists()
    
    def test_clear_cache(self):
        """Test cache clearing."""
        executor = NomaExecutor()
        
        # Create a test cache file
        test_cache = executor.cache_dir / "test_cache.txt"
        test_cache.write_text("test")
        assert test_cache.exists()
        
        # Clear cache
        executor.clear_cache()
        
        # Cache should still exist as directory but be empty
        assert executor.cache_dir.exists()
        assert not test_cache.exists()
    
    def test_list_artifacts(self):
        """Test listing artifacts."""
        executor = NomaExecutor()
        artifacts = executor.list_artifacts()
        
        assert "workspace" in artifacts
        assert "cache" in artifacts
        assert "outputs" in artifacts
        assert isinstance(artifacts["workspace"], list)


class TestMagicIntegration:
    """Test integration with IPython (mock)."""
    
    def test_load_extension(self):
        """Test that extension can be loaded."""
        from noma_magic import load_ipython_extension
        
        # Create mock IPython instance
        mock_ipython = MagicMock()
        
        # Should not raise
        load_ipython_extension(mock_ipython)
        
        # Should register magics
        mock_ipython.register_magics.assert_called_once()


if __name__ == "__main__":
    # Simple test runner
    test_exec = TestNomaExecutor()
    
    print("Testing NomaExecutor...")
    test_exec.test_init()
    print("[PASS] test_init")
    
    test_exec.test_hash_content()
    print("[PASS] test_hash_content")
    
    test_exec.test_cache_path()
    print("[PASS] test_cache_path")
    
    test_exec.test_save_execution_log()
    print("[PASS] test_save_execution_log")
    
    test_exec.test_clear_cache()
    print("[PASS] test_clear_cache")
    
    test_exec.test_list_artifacts()
    print("[PASS] test_list_artifacts")
    
    test_magic = TestMagicIntegration()
    test_magic.test_load_extension()
    print("[PASS] test_load_extension")
    
    print("\nAll tests passed!")
