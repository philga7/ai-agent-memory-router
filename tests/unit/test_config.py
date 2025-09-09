"""
Unit tests for configuration management
Adapted from test-automation-harness patterns
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from pydantic import ValidationError
from app.core.config import get_settings, Settings, DatabaseSettings, CipherSettings, WeaviateSettings


class TestConfiguration:
    """Test configuration management functionality."""
    
    def test_settings_initialization(self):
        """Test that settings are properly initialized."""
        settings = get_settings()
        
        assert settings is not None
        assert hasattr(settings, 'database')
        assert hasattr(settings, 'cipher')
        assert hasattr(settings, 'weaviate')
        assert hasattr(settings, 'security')
        assert hasattr(settings, 'environment')
        assert hasattr(settings, 'log_level')
    
    def test_settings_default_values(self):
        """Test that settings have appropriate default values."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()
            
            # Test default values
            assert settings.security.algorithm == "HS256"
            assert settings.security.access_token_expire_minutes == 30
            assert settings.environment == "development"
    
    def test_settings_environment_override(self):
        """Test that environment variables override defaults."""
        test_env = {
            "DATABASE_URL": "sqlite:///test.db",
            "SECRET_KEY": "test-secret-key",
            "ALGORITHM": "HS512",
            "ACCESS_TOKEN_EXPIRE_MINUTES": "60",
            "ENVIRONMENT": "production"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()
            
            assert settings.database.url == "sqlite:///test.db"
            assert settings.security.secret_key == "test-secret-key"
            assert settings.security.algorithm == "HS512"
            assert settings.security.access_token_expire_minutes == 60
            assert settings.environment == "production"
    
    def test_settings_validation(self):
        """Test settings validation."""
        # Test valid settings
        valid_settings = {
            "environment": "development",
            "log_level": "DEBUG"
        }
        
        settings = Settings(**valid_settings)
        assert settings.environment == "development"
        assert settings.log_level == "DEBUG"
    
    def test_settings_invalid_values(self):
        """Test settings validation with invalid values."""
        # Test invalid environment
        with pytest.raises(ValidationError):
            Settings(environment="invalid_environment")
        
        # Test invalid log level
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID_LEVEL")
    
    def test_settings_type_conversion(self):
        """Test that settings properly convert types."""
        test_env = {
            "ACCESS_TOKEN_EXPIRE_MINUTES": "45",
            "API_PORT": "9000",
            "LOG_LEVEL": "DEBUG"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()
            
            assert isinstance(settings.security.access_token_expire_minutes, int)
            assert settings.security.access_token_expire_minutes == 45
            assert isinstance(settings.api_port, int)
            assert settings.api_port == 9000
    
    def test_settings_singleton_pattern(self):
        """Test that get_settings returns the same instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
    
    def test_settings_with_missing_required_fields(self):
        """Test settings behavior with missing required fields."""
        # Test with minimal required fields
        minimal_settings = {
            "environment": "development"
        }
        
        settings = Settings(**minimal_settings)
        assert settings.environment == "development"
        # Should use defaults for other fields
        assert settings.security.algorithm == "HS256"
    
    def test_settings_database_url_validation(self):
        """Test database URL validation."""
        # Test valid database URLs
        valid_urls = [
            "sqlite:///test.db",
            "postgresql://user:pass@localhost/db",
            "mysql://user:pass@localhost/db"
        ]
        
        for url in valid_urls:
            settings = Settings(database=DatabaseSettings(url=url))
            assert settings.database.url == url
    
    def test_settings_cipher_api_url_validation(self):
        """Test Cipher API URL validation."""
        valid_urls = [
            "http://localhost:3001",
            "https://api.cipher.com",
            "http://127.0.0.1:8080"
        ]
        
        for url in valid_urls:
            settings = Settings(cipher=CipherSettings(api_url=url))
            assert settings.cipher.api_url == url
    
    def test_settings_weaviate_url_validation(self):
        """Test Weaviate URL validation."""
        valid_urls = [
            "http://localhost:8080",
            "https://weaviate.example.com",
            "http://127.0.0.1:8080"
        ]
        
        for url in valid_urls:
            settings = Settings(weaviate=WeaviateSettings(api_url=url))
            assert settings.weaviate.api_url == url
    
    def test_settings_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_levels:
            settings = Settings(log_level=level)
            assert settings.log_level == level
    
    def test_settings_environment_variable_priority(self):
        """Test that environment variables take priority over defaults."""
        test_env = {
            "SECRET_KEY": "env-secret-key",
            "ALGORITHM": "HS512",
            "DATABASE_URL": "env-database-url"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()
            
            assert settings.security.secret_key == "env-secret-key"
            assert settings.security.algorithm == "HS512"
            assert settings.database.url == "env-database-url"
    
    def test_settings_mutability(self):
        """Test that settings can be modified after creation (Pydantic v2 behavior)."""
        settings = Settings(environment="development")
        
        # In Pydantic v2, settings are mutable by default
        settings.environment = "production"
        assert settings.environment == "production"
    
    def test_settings_serialization(self):
        """Test settings serialization."""
        settings = Settings(
            environment="development",
            log_level="DEBUG"
        )
        
        # Test dict conversion
        settings_dict = settings.model_dump()
        assert isinstance(settings_dict, dict)
        assert settings_dict["environment"] == "development"
        assert settings_dict["log_level"] == "DEBUG"
        
        # Test JSON serialization
        settings_json = settings.model_dump_json()
        assert isinstance(settings_json, str)
        assert "development" in settings_json
    
    def test_settings_with_testing_mode(self):
        """Test settings behavior in testing mode."""
        test_env = {
            "ENVIRONMENT": "development",
            "SECRET_KEY": "test-secret"
        }
        
        with patch.dict(os.environ, test_env, clear=True):
            settings = Settings()
            
            assert settings.environment == "development"
            # In development mode, some settings might behave differently
            assert settings.security.secret_key == "test-secret"
