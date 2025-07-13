#!/usr/bin/env python3
"""
Test suite for Kimi K2 Code Agent

Tests cover:
- Code request detection
- Context extraction
- API integration
- Diff generation
- Error handling
- Configuration management
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.kimi_agent import (
    KimiK2Agent, KimiConfig, CodeContext, CodeResponse,
    CodeContextExtractor, DiffGenerator, create_kimi_agent
)


class TestKimiConfig:
    """Test KimiConfig configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = KimiConfig()
        
        assert config.model_id == "kimi-k2"
        assert config.base_url == "https://api.moonshot.cn/v1"
        assert config.temperature == 0.1
        assert config.max_tokens == 4000
        assert config.timeout == 60.0
        assert 'python' in config.supported_languages
        assert '*.py' in config.include_patterns
        assert '__pycache__/*' in config.exclude_patterns
    
    def test_config_with_overrides(self):
        """Test configuration with custom values"""
        config = KimiConfig(
            api_key="test-key",
            model_id="custom-kimi",
            temperature=0.2,
            max_tokens=2000
        )
        
        assert config.api_key == "test-key"
        assert config.model_id == "custom-kimi"
        assert config.temperature == 0.2
        assert config.max_tokens == 2000


class TestCodeContext:
    """Test CodeContext data structure"""
    
    def test_empty_context(self):
        """Test empty code context"""
        context = CodeContext()
        
        assert context.current_file is None
        assert context.current_file_content is None
        assert context.project_files == []
        assert context.dependencies == []
    
    def test_context_with_data(self):
        """Test code context with data"""
        context = CodeContext(
            current_file="test.py",
            current_file_content="print('hello')",
            language="python",
            project_files=[{"path": "main.py", "content": "# main"}]
        )
        
        assert context.current_file == "test.py"
        assert context.language == "python"
        assert len(context.project_files) == 1


class TestCodeResponse:
    """Test CodeResponse data structure"""
    
    def test_basic_response(self):
        """Test basic code response"""
        response = CodeResponse(
            content="def hello(): return 'Hello World'",
            operation_type="generate",
            language="python"
        )
        
        assert "def hello()" in response.content
        assert response.operation_type == "generate"
        assert response.language == "python"
        assert response.execution_time == 0.0


class TestDiffGenerator:
    """Test DiffGenerator functionality"""
    
    def test_unified_diff_generation(self):
        """Test unified diff generation"""
        original = "def hello():\n    return 'Hello'"
        modified = "def hello():\n    return 'Hello World'"
        
        diff = DiffGenerator.generate_unified_diff(original, modified, "test.py")
        
        assert "@@" in diff  # Unified diff format
        assert "-    return 'Hello'" in diff
        assert "+    return 'Hello World'" in diff
    
    def test_diff_formatting_for_cursor(self):
        """Test diff formatting for Cursor IDE"""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def hello():
-    return 'Hello'
+    return 'Hello World'"""
        
        formatted = DiffGenerator.format_diff_for_cursor(diff)
        
        assert "📁" in formatted  # File indicators
        assert "📍" in formatted  # Location indicators
        assert "✅" in formatted  # Additions
        assert "❌" in formatted  # Deletions
    
    def test_empty_diff(self):
        """Test diff generation with identical content"""
        content = "def hello(): pass"
        diff = DiffGenerator.generate_unified_diff(content, content, "test.py")
        
        # Should be empty or minimal diff
        assert len(diff.strip()) == 0 or "@@" not in diff


class TestCodeContextExtractor:
    """Test CodeContextExtractor functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = KimiConfig()
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = CodeContextExtractor(self.config)
        # Override project root for testing
        self.extractor.project_root = Path(self.temp_dir)
    
    def teardown_method(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_language_detection(self):
        """Test programming language detection"""
        assert self.extractor._detect_language("test.py") == "python"
        assert self.extractor._detect_language("app.js") == "javascript"
        assert self.extractor._detect_language("main.cpp") == "cpp"
        assert self.extractor._detect_language("style.css") == "css"
        assert self.extractor._detect_language("unknown.xyz") is None
    
    def test_file_reading(self):
        """Test safe file reading"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.py"
        test_content = "print('Hello World')"
        test_file.write_text(test_content)
        
        content = self.extractor._read_file_safe(str(test_file))
        assert content == test_content
    
    def test_file_reading_nonexistent(self):
        """Test reading non-existent file"""
        content = self.extractor._read_file_safe("nonexistent.py")
        assert content is None
    
    def test_should_include_file(self):
        """Test file inclusion logic"""
        # Create test files
        good_file = Path(self.temp_dir) / "main.py"
        bad_file = Path(self.temp_dir) / "__pycache__" / "cache.pyc"
        
        good_file.write_text("print('hello')")
        bad_file.parent.mkdir(exist_ok=True)
        bad_file.write_bytes(b'\x00\x01\x02')  # Binary content
        
        assert self.extractor._should_include_file(good_file) == True
        assert self.extractor._should_include_file(bad_file) == False
    
    def test_project_files_extraction(self):
        """Test project files extraction"""
        # Create test files
        (Path(self.temp_dir) / "main.py").write_text("# Main file")
        (Path(self.temp_dir) / "utils.py").write_text("# Utils")
        (Path(self.temp_dir) / "README.md").write_text("# README")
        
        files = self.extractor._get_relevant_project_files()
        
        assert len(files) >= 2  # At least main.py and utils.py
        assert any(f['name'] == 'main.py' for f in files)
        assert any(f['language'] == 'python' for f in files)
    
    def test_dependency_extraction(self):
        """Test dependency extraction"""
        # Create requirements.txt
        req_file = Path(self.temp_dir) / "requirements.txt"
        req_file.write_text("requests>=2.28.0\nnumpy>=1.20.0")
        
        dependencies = self.extractor._extract_dependencies()
        
        assert len(dependencies) > 0
        assert any("requirements.txt" in dep for dep in dependencies)
        assert any("requests" in dep for dep in dependencies)
    
    def test_context_extraction_with_file(self):
        """Test complete context extraction with current file"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.py"
        test_file.write_text("def hello():\n    return 'world'")
        
        context = self.extractor.extract_context(
            current_file=str(test_file),
            include_git=False,
            include_errors=False
        )
        
        assert context.current_file == str(test_file)
        assert "def hello()" in context.current_file_content
        assert context.language == "python"
    
    def test_context_extraction_no_file(self):
        """Test context extraction without current file"""
        context = self.extractor.extract_context(
            include_git=False,
            include_errors=False
        )
        
        assert context.current_file is None
        assert context.current_file_content is None
        assert isinstance(context.project_files, list)


class TestKimiK2Agent:
    """Test KimiK2Agent main functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = KimiConfig(api_key="test-key")
        self.agent = KimiK2Agent(self.config)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        assert self.agent.config.api_key == "test-key"
        assert isinstance(self.agent.context_extractor, CodeContextExtractor)
        assert isinstance(self.agent.diff_generator, DiffGenerator)
        assert self.agent.client is None  # Not initialized yet
    
    def test_default_config_creation(self):
        """Test default configuration with environment variables"""
        with patch.dict(os.environ, {
            'KIMI_API_KEY': 'env-key',
            'KIMI_BASE_URL': 'https://custom.api.com',
            'KIMI_MODEL_ID': 'custom-model'
        }):
            agent = KimiK2Agent()
            
            assert agent.config.api_key == 'env-key'
            assert agent.config.base_url == 'https://custom.api.com'
            assert agent.config.model_id == 'custom-model'
    
    def test_moonshot_api_key_fallback(self):
        """Test fallback to MOONSHOT_API_KEY"""
        with patch.dict(os.environ, {'MOONSHOT_API_KEY': 'moonshot-key'}, clear=True):
            agent = KimiK2Agent()
            assert agent.config.api_key == 'moonshot-key'
    
    def test_code_request_detection_explicit(self):
        """Test explicit code request detection with #code tag"""
        assert self.agent.detect_code_request("#code generate a hello world function") == True
        assert self.agent.detect_code_request("#code debug this error") == True
        assert self.agent.detect_code_request("  #code   refactor my class") == True
    
    def test_code_request_detection_keywords(self):
        """Test code request detection via keywords"""
        assert self.agent.detect_code_request("write a function to sort numbers") == True
        assert self.agent.detect_code_request("debug this error in my code") == True
        assert self.agent.detect_code_request("refactor my implementation") == True
        assert self.agent.detect_code_request("explain this algorithm") == True
        assert self.agent.detect_code_request("optimize the performance") == True
        assert self.agent.detect_code_request("how is the weather today?") == False
        assert self.agent.detect_code_request("tell me a joke") == False
    
    def test_code_request_extraction(self):
        """Test code request extraction and operation classification"""
        # Test explicit #code tag removal
        operation, clean = self.agent.extract_code_request("#code generate a sorting function")
        assert operation == "generate"
        assert clean == "generate a sorting function"
        
        # Test operation detection
        operation, clean = self.agent.extract_code_request("debug this error message")
        assert operation == "debug"
        assert clean == "debug this error message"
        
        operation, clean = self.agent.extract_code_request("refactor my messy code")
        assert operation == "refactor"
        
        operation, clean = self.agent.extract_code_request("explain how this works")
        assert operation == "explain"
        
        # Test default to generate
        operation, clean = self.agent.extract_code_request("create something cool")
        assert operation == "generate"
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_api_key(self):
        """Test async agent initialization"""
        await self.agent.initialize()
        
        assert self.agent.client is not None
        assert self.agent.client.headers['Authorization'] == 'Bearer test-key'
        assert self.agent.client.headers['Content-Type'] == 'application/json'
        
        await self.agent.cleanup()
    
    @pytest.mark.asyncio
    async def test_agent_initialization_no_api_key(self):
        """Test agent initialization failure without API key"""
        agent = KimiK2Agent(KimiConfig())  # No API key
        
        with pytest.raises(ValueError, match="Kimi API key is required"):
            await agent.initialize()
    
    @pytest.mark.asyncio
    async def test_agent_cleanup(self):
        """Test agent cleanup"""
        await self.agent.initialize()
        assert self.agent.client is not None
        
        await self.agent.cleanup()
        assert self.agent.client is None
    
    @pytest.mark.asyncio
    async def test_code_request_processing_mock(self):
        """Test code request processing with mocked API"""
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [
                {
                    'message': {
                        'content': 'def hello_world():\n    return "Hello, World!"'
                    }
                }
            ],
            'usage': {
                'total_tokens': 50,
                'prompt_tokens': 20,
                'completion_tokens': 30
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        # Initialize agent and replace client
        await self.agent.initialize()
        self.agent.client = mock_client
        
        # Process request
        response = await self.agent.process_code_request(
            message="#code generate a hello world function in Python",
            current_file=None,
            error_context=None
        )
        
        # Verify response
        assert response.operation_type == "generate"
        assert "def hello_world()" in response.content
        assert response.token_usage is not None
        assert response.execution_time > 0
        
        # Verify API call
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/chat/completions" in call_args[0][0]
        
        payload = call_args[1]['json']
        assert payload['model'] == 'kimi-k2'
        assert len(payload['messages']) >= 2
        assert payload['temperature'] == 0.1
        
        await self.agent.cleanup()
    
    def test_system_prompt_building(self):
        """Test system prompt construction"""
        context = CodeContext(language="python")
        
        # Test different operations
        prompt = self.agent._build_system_prompt("generate", context)
        assert "Generate clean, well-documented" in prompt
        assert "python" in prompt
        
        prompt = self.agent._build_system_prompt("debug", context)
        assert "Analyze the code and error context" in prompt
        
        prompt = self.agent._build_system_prompt("refactor", context)
        assert "Improve the provided code" in prompt
    
    def test_user_prompt_building(self):
        """Test user prompt construction with context"""
        context = CodeContext(
            current_file="test.py",
            current_file_content="print('hello')",
            language="python",
            project_files=[{"path": "main.py", "content": "# main"}],
            git_status="M test.py",
            dependencies=["requirements.txt:\nrequests>=2.0"],
            error_content="NameError: name 'foo' is not defined"
        )
        
        prompt = self.agent._build_user_prompt("Fix this error", context)
        
        assert "Fix this error" in prompt
        assert "test.py" in prompt
        assert "print('hello')" in prompt
        assert "NameError" in prompt
        assert "Project structure" in prompt
        assert "Git status" in prompt
        assert "requirements.txt" in prompt
    
    def test_code_extraction_from_response(self):
        """Test code block extraction from LLM response"""
        response_with_code = """
        Here's the solution:
        
        ```python
        def hello_world():
            return "Hello, World!"
        ```
        
        This function returns a greeting.
        """
        
        code = self.agent._extract_code_from_response(response_with_code)
        assert code is not None
        assert "def hello_world()" in code
        assert "return \"Hello, World!\"" in code
        
        # Test response without code blocks
        response_no_code = "This is just text without any code blocks."
        code = self.agent._extract_code_from_response(response_no_code)
        assert code is None
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        stats = self.agent.get_stats()
        
        assert 'performance' in stats
        assert 'configuration' in stats
        assert 'context_info' in stats
        
        assert stats['performance']['total_requests'] == 0
        assert stats['performance']['successful_requests'] == 0
        assert stats['performance']['failed_requests'] == 0
        
        # Test reset
        self.agent.stats['total_requests'] = 5
        self.agent.reset_stats()
        assert self.agent.stats['total_requests'] == 0


class TestFactoryFunction:
    """Test factory functions"""
    
    def test_create_kimi_agent(self):
        """Test create_kimi_agent factory function"""
        agent = create_kimi_agent()
        assert isinstance(agent, KimiK2Agent)
        assert isinstance(agent.config, KimiConfig)
        
        # Test with custom config
        custom_config = KimiConfig(api_key="custom-key")
        agent = create_kimi_agent(custom_config)
        assert agent.config.api_key == "custom-key"


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_mock(self):
        """Test complete workflow with mocked components"""
        # Create agent with test config
        config = KimiConfig(api_key="test-key")
        agent = KimiK2Agent(config)
        
        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{'message': {'content': 'def sort_list(lst):\n    return sorted(lst)'}}],
            'usage': {'total_tokens': 42}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        
        # Initialize and replace client
        await agent.initialize()
        agent.client = mock_client
        
        # Test the complete workflow
        request = "#code generate a function to sort a list in Python"
        
        # Step 1: Detection
        assert agent.detect_code_request(request) == True
        
        # Step 2: Extraction
        operation, clean_request = agent.extract_code_request(request)
        assert operation == "generate"
        
        # Step 3: Processing
        response = await agent.process_code_request(request)
        
        # Verify results
        assert response.operation_type == "generate"
        assert "def sort_list" in response.content
        assert response.execution_time > 0
        
        # Verify statistics
        stats = agent.get_stats()
        assert stats['performance']['total_requests'] == 1
        assert stats['performance']['successful_requests'] == 1
        assert stats['performance']['operations']['generate'] == 1
        
        await agent.cleanup()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v']) 