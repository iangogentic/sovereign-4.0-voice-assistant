"""
Test development environment setup and basic functionality
"""

import os
import sys
import yaml
import pytest
from pathlib import Path

# Add the parent directory to sys.path to import assistant
sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant import __version__


class TestDevelopmentEnvironment:
    """Test that the development environment is set up correctly"""
    
    def test_version_is_defined(self):
        """Test that the package version is defined"""
        assert __version__ is not None
        assert __version__ == "0.1.0"
    
    def test_project_structure(self):
        """Test that all required directories exist"""
        project_root = Path(__file__).parent.parent
        
        # Check required directories
        required_dirs = [
            "assistant",
            "config",
            "tests",
            "docs",
            ".taskmaster"
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            assert dir_path.exists(), f"Required directory {dir_name} does not exist"
            assert dir_path.is_dir(), f"{dir_name} is not a directory"
    
    def test_configuration_files(self):
        """Test that configuration files exist and are valid"""
        project_root = Path(__file__).parent.parent
        
        # Check requirements.txt
        requirements_file = project_root / "requirements.txt"
        assert requirements_file.exists(), "requirements.txt does not exist"
        
        # Check config.yaml
        config_file = project_root / "config" / "config.yaml"
        assert config_file.exists(), "config/config.yaml does not exist"
        
        # Validate YAML syntax
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required config sections
        required_sections = [
            "audio",
            "stt",
            "tts",
            "llm",
            "memory",
            "screen_monitor",
            "features"
        ]
        
        for section in required_sections:
            assert section in config, f"Required config section '{section}' is missing"
    
    def test_environment_template(self):
        """Test that .env.example exists and contains required variables"""
        project_root = Path(__file__).parent.parent
        env_example = project_root / ".env.example"
        
        assert env_example.exists(), ".env.example does not exist"
        
        # Check for required environment variables
        with open(env_example, 'r') as f:
            content = f.read()
        
        required_vars = [
            "OPENAI_API_KEY",
            "KIMI_API_KEY",
            "CONFIG_PATH"
        ]
        
        for var in required_vars:
            assert var in content, f"Required environment variable {var} not found in .env.example"
    
    def test_gitignore_setup(self):
        """Test that .gitignore is properly configured"""
        project_root = Path(__file__).parent.parent
        gitignore_file = project_root / ".gitignore"
        
        assert gitignore_file.exists(), ".gitignore does not exist"
        
        with open(gitignore_file, 'r') as f:
            content = f.read()
        
        # Check for important ignores (more flexible patterns)
        important_ignores = [
            "__pycache__",
            "*.py[cod]",  # This covers *.pyc as well
            ".env",
            "logs/",
            "models/",
            "data/"
        ]
        
        for ignore in important_ignores:
            assert ignore in content, f"Important ignore pattern '{ignore}' not found in .gitignore"
    
    def test_import_assistant_package(self):
        """Test that the assistant package can be imported"""
        try:
            import assistant
            assert hasattr(assistant, '__version__')
        except ImportError as e:
            pytest.fail(f"Failed to import assistant package: {e}")
    
    def test_required_dependencies_listed(self):
        """Test that all required dependencies are listed in requirements.txt"""
        project_root = Path(__file__).parent.parent
        requirements_file = project_root / "requirements.txt"
        
        with open(requirements_file, 'r') as f:
            content = f.read()
        
        # Check for core dependencies
        required_deps = [
            "pipecat-ai",
            "openai",
            "chromadb",
            "langchain",
            "pytesseract",
            "mss",
            "pyyaml"
        ]
        
        for dep in required_deps:
            assert dep in content, f"Required dependency '{dep}' not found in requirements.txt"


if __name__ == "__main__":
    pytest.main([__file__]) 