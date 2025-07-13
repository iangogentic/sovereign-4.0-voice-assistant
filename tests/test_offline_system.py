#!/usr/bin/env python3
"""
Test suite for Offline Fallback System

Tests cover:
- Network connectivity detection
- Model management and downloading
- Memory monitoring and pressure handling
- Offline STT, TTS, and LLM services
- Complete offline system integration
- Performance and reliability
"""

import asyncio
import pytest
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import numpy as np
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant.offline_system import (
    OfflineSystem, OfflineConfig, NetworkDetector, ModelManager, MemoryMonitor,
    OfflineSTTService, OfflineTTSService, OfflineLLMService,
    ConnectivityStatus, ModelStatus, ModelInfo, ConnectivityResult,
    create_offline_system
)


class TestOfflineConfig:
    """Test OfflineConfig configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = OfflineConfig()
        
        assert config.models_dir == "./data/offline_models"
        assert config.whisper_model == "tiny.en"
        assert config.piper_voice == "en_US-lessac-medium"
        assert config.llama_model == "gemma-2b-it-q4_k_m.gguf"
        assert config.target_offline_latency == 1.5
        assert config.max_memory_usage_percent == 85.0
        assert "8.8.8.8" in config.ping_hosts
        assert "1.1.1.1" in config.ping_hosts
    
    def test_config_customization(self):
        """Test configuration customization"""
        config = OfflineConfig(
            models_dir="/custom/path",
            whisper_model="base.en",
            target_offline_latency=2.0,
            max_memory_usage_percent=90.0
        )
        
        assert config.models_dir == "/custom/path"
        assert config.whisper_model == "base.en"
        assert config.target_offline_latency == 2.0
        assert config.max_memory_usage_percent == 90.0


class TestNetworkDetector:
    """Test NetworkDetector connectivity detection"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OfflineConfig()
        self.detector = NetworkDetector(self.config)
    
    def test_initialization(self):
        """Test NetworkDetector initialization"""
        assert self.detector.config == self.config
        assert self.detector.consecutive_failures == 0
        assert self.detector.last_result.status == ConnectivityStatus.UNKNOWN
    
    @patch('subprocess.run')
    def test_ping_connectivity_success(self, mock_run):
        """Test successful ping connectivity check"""
        mock_run.return_value = Mock(returncode=0)
        
        result = self.detector.check_ping_connectivity()
        
        assert result.status == ConnectivityStatus.ONLINE
        assert result.latency_ms > 0
        assert result.error is None
    
    @patch('subprocess.run')
    def test_ping_connectivity_failure(self, mock_run):
        """Test failed ping connectivity check"""
        mock_run.return_value = Mock(returncode=1)
        
        result = self.detector.check_ping_connectivity()
        
        assert result.status == ConnectivityStatus.OFFLINE
        assert result.latency_ms == 0
        assert result.error == "All ping hosts unreachable"
    
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_api_connectivity_success(self, mock_get):
        """Test successful API connectivity check"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = await self.detector.check_api_connectivity()
        
        assert result.status == ConnectivityStatus.ONLINE
        assert result.latency_ms > 0
        assert result.error is None
    
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_api_connectivity_failure(self, mock_get):
        """Test failed API connectivity check"""
        mock_get.side_effect = Exception("Connection failed")
        
        result = await self.detector.check_api_connectivity()
        
        assert result.status == ConnectivityStatus.OFFLINE
        assert result.latency_ms == 0
        assert result.error == "All API endpoints unreachable"
    
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_api_connectivity_degraded(self, mock_get):
        """Test degraded API connectivity (auth errors still count as connected)"""
        mock_response = Mock()
        mock_response.status_code = 401  # Auth error, but connection works
        mock_get.return_value = mock_response
        
        result = await self.detector.check_api_connectivity()
        
        assert result.status == ConnectivityStatus.ONLINE
        assert result.latency_ms > 0
    
    @patch.object(NetworkDetector, 'check_ping_connectivity')
    @patch.object(NetworkDetector, 'check_api_connectivity')
    @pytest.mark.asyncio
    async def test_comprehensive_connectivity_online(self, mock_api, mock_ping):
        """Test comprehensive connectivity check - online"""
        mock_ping.return_value = ConnectivityResult(
            status=ConnectivityStatus.ONLINE,
            latency_ms=10.0,
            timestamp=time.time()
        )
        mock_api.return_value = ConnectivityResult(
            status=ConnectivityStatus.ONLINE,
            latency_ms=50.0,
            timestamp=time.time()
        )
        
        result = await self.detector.check_connectivity()
        
        assert result.status == ConnectivityStatus.ONLINE
        assert result.latency_ms == 30.0  # Average of ping and API
        assert self.detector.consecutive_failures == 0
    
    @patch.object(NetworkDetector, 'check_ping_connectivity')
    @patch.object(NetworkDetector, 'check_api_connectivity')
    @pytest.mark.asyncio
    async def test_comprehensive_connectivity_degraded(self, mock_api, mock_ping):
        """Test comprehensive connectivity check - degraded"""
        mock_ping.return_value = ConnectivityResult(
            status=ConnectivityStatus.ONLINE,
            latency_ms=10.0,
            timestamp=time.time()
        )
        mock_api.return_value = ConnectivityResult(
            status=ConnectivityStatus.OFFLINE,
            latency_ms=0.0,
            timestamp=time.time()
        )
        
        result = await self.detector.check_connectivity()
        
        assert result.status == ConnectivityStatus.DEGRADED
        assert result.latency_ms == 10.0
        assert result.error == "API endpoints unreachable"
    
    @pytest.mark.asyncio
    async def test_connectivity_caching(self):
        """Test that connectivity results are cached"""
        # Mock the actual check methods to ensure they're not called
        with patch.object(self.detector, 'check_ping_connectivity') as mock_ping:
            mock_ping.return_value = ConnectivityResult(
                status=ConnectivityStatus.ONLINE,
                latency_ms=10.0,
                timestamp=time.time()
            )
            
            # First call should check
            result1 = await self.detector.check_connectivity()
            assert mock_ping.called
            
            # Second call should use cache
            mock_ping.reset_mock()
            result2 = await self.detector.check_connectivity()
            assert not mock_ping.called
            assert result1.status == result2.status


class TestModelManager:
    """Test ModelManager model downloading and management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OfflineConfig(models_dir=self.temp_dir)
        self.manager = ModelManager(self.config)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test ModelManager initialization"""
        assert self.manager.config == self.config
        assert len(self.manager.models) >= 3  # whisper, piper, gemma
        assert "whisper_tiny_en" in self.manager.models
        assert "piper_lessac_medium" in self.manager.models
        assert "gemma_2b_q4" in self.manager.models
    
    def test_model_info_structure(self):
        """Test ModelInfo structure"""
        whisper_model = self.manager.models["whisper_tiny_en"]
        
        assert isinstance(whisper_model, ModelInfo)
        assert whisper_model.name == "whisper_tiny_en"
        assert whisper_model.url.startswith("https://")
        assert whisper_model.size_mb > 0
        assert whisper_model.status == ModelStatus.NOT_LOADED
    
    def test_file_checksum_calculation(self):
        """Test file checksum calculation"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        checksum = self.manager.calculate_file_checksum(str(test_file))
        
        assert isinstance(checksum, str)
        assert len(checksum) == 40  # SHA-1 hex length
    
    def test_model_verification_missing_file(self):
        """Test model verification with missing file"""
        model_info = ModelInfo(
            name="test_model",
            url="https://example.com/model",
            file_path=str(Path(self.temp_dir) / "missing.bin"),
            size_mb=100.0,
            checksum="abc123"
        )
        
        assert not self.manager.verify_model(model_info)
    
    def test_model_verification_size_mismatch(self):
        """Test model verification with size mismatch"""
        # Create a test file
        test_file = Path(self.temp_dir) / "test.bin"
        test_file.write_bytes(b"small content")  # Much smaller than expected
        
        model_info = ModelInfo(
            name="test_model",
            url="https://example.com/model",
            file_path=str(test_file),
            size_mb=100.0,  # Expect 100MB but file is tiny
            checksum="abc123"
        )
        
        assert not self.manager.verify_model(model_info)
    
    def test_model_verification_success(self):
        """Test successful model verification"""
        # Create a test file with appropriate size
        test_file = Path(self.temp_dir) / "test.bin"
        test_content = b"x" * (50 * 1024 * 1024)  # 50MB
        test_file.write_bytes(test_content)
        
        model_info = ModelInfo(
            name="test_model",
            url="https://example.com/model",
            file_path=str(test_file),
            size_mb=50.0,
            checksum="abc123"
        )
        
        assert self.manager.verify_model(model_info)
    
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_model_download_success(self, mock_get):
        """Test successful model download"""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content = Mock(return_value=[b'x' * 1000])
        mock_get.return_value = mock_response
        
        # Create test model info
        model_info = ModelInfo(
            name="test_model",
            url="https://example.com/model",
            file_path=str(Path(self.temp_dir) / "test.bin"),
            size_mb=0.001,  # 1KB
            checksum="abc123"
        )
        
        # Mock verification to return True
        with patch.object(self.manager, 'verify_model', return_value=True):
            result = await self.manager.download_model(model_info)
        
        assert result is True
        assert model_info.status == ModelStatus.LOADED
        assert Path(model_info.file_path).exists()
    
    @patch('requests.get')
    @pytest.mark.asyncio
    async def test_model_download_failure(self, mock_get):
        """Test failed model download"""
        mock_get.side_effect = Exception("Download failed")
        
        model_info = ModelInfo(
            name="test_model",
            url="https://example.com/model",
            file_path=str(Path(self.temp_dir) / "test.bin"),
            size_mb=0.001,
            checksum="abc123"
        )
        
        result = await self.manager.download_model(model_info)
        
        assert result is False
        assert model_info.status == ModelStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_ensure_models_available_existing(self):
        """Test ensuring models are available when they already exist"""
        # Create a fake model file
        model_file = Path(self.temp_dir) / "ggml-tiny.en.bin"
        model_file.write_bytes(b"x" * (39 * 1024 * 1024))  # 39MB
        
        results = await self.manager.ensure_models_available(["whisper_tiny_en"])
        
        assert results["whisper_tiny_en"] is True
        assert self.manager.models["whisper_tiny_en"].status == ModelStatus.LOADED
    
    def test_get_model_path_existing(self):
        """Test getting model path for existing model"""
        # Create a fake model file
        model_file = Path(self.temp_dir) / "ggml-tiny.en.bin"
        model_file.write_bytes(b"x" * (39 * 1024 * 1024))  # 39MB
        
        path = self.manager.get_model_path("whisper_tiny_en")
        
        assert path == str(model_file)
    
    def test_get_model_path_missing(self):
        """Test getting model path for missing model"""
        path = self.manager.get_model_path("nonexistent_model")
        assert path is None
    
    def test_get_models_status(self):
        """Test getting models status"""
        status = self.manager.get_models_status()
        
        assert isinstance(status, dict)
        assert "whisper_tiny_en" in status
        assert "status" in status["whisper_tiny_en"]
        assert "size_mb" in status["whisper_tiny_en"]
        assert "file_exists" in status["whisper_tiny_en"]
        assert "verified" in status["whisper_tiny_en"]


class TestMemoryMonitor:
    """Test MemoryMonitor memory management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OfflineConfig()
        self.monitor = MemoryMonitor(self.config)
    
    def test_initialization(self):
        """Test MemoryMonitor initialization"""
        assert self.monitor.config == self.config
        assert self.monitor.last_check == 0.0
        assert len(self.monitor.memory_pressure_callbacks) == 0
    
    def test_get_memory_info(self):
        """Test getting memory information"""
        memory_info = self.monitor.get_memory_info()
        
        assert isinstance(memory_info, dict)
        assert "total_gb" in memory_info
        assert "available_gb" in memory_info
        assert "used_gb" in memory_info
        assert "percent_used" in memory_info
        assert "free_gb" in memory_info
        
        # Basic sanity checks
        assert memory_info["total_gb"] > 0
        assert memory_info["percent_used"] >= 0
        assert memory_info["percent_used"] <= 100
    
    @patch('psutil.virtual_memory')
    def test_check_memory_pressure_high(self, mock_memory):
        """Test memory pressure detection when pressure is high"""
        mock_memory.return_value = Mock(
            percent=90.0,
            total=8589934592,  # 8GB
            available=858993459,  # 0.8GB
            used=7730941133,  # 7.2GB
            free=858993459  # 0.8GB
        )
        
        assert self.monitor.check_memory_pressure() is True
    
    @patch('psutil.virtual_memory')
    def test_check_memory_pressure_low(self, mock_memory):
        """Test memory pressure detection when pressure is low"""
        mock_memory.return_value = Mock(
            percent=70.0,
            total=8589934592,  # 8GB
            available=2576980378,  # 2.4GB
            used=6012954214,  # 5.6GB
            free=2576980378  # 2.4GB
        )
        
        assert self.monitor.check_memory_pressure() is False
    
    @patch('psutil.virtual_memory')
    def test_should_unload_models(self, mock_memory):
        """Test model unloading threshold"""
        mock_memory.return_value = Mock(
            percent=95.0,
            total=8589934592,  # 8GB
            available=429496729,  # 0.4GB
            used=8160437863,  # 7.6GB
            free=429496729  # 0.4GB
        )
        
        assert self.monitor.should_unload_models() is True
    
    def test_add_pressure_callback(self):
        """Test adding memory pressure callback"""
        callback = Mock()
        
        self.monitor.add_pressure_callback(callback)
        
        assert len(self.monitor.memory_pressure_callbacks) == 1
        assert callback in self.monitor.memory_pressure_callbacks
    
    @patch('psutil.virtual_memory')
    @pytest.mark.asyncio
    async def test_memory_monitoring_with_pressure(self, mock_memory):
        """Test memory monitoring with pressure detection"""
        mock_memory.return_value = Mock(
            percent=90.0,
            total=8589934592,  # 8GB
            available=858993459,  # 0.8GB
            used=7730941133,  # 7.2GB
            free=858993459  # 0.8GB
        )
        
        callback = AsyncMock()
        self.monitor.add_pressure_callback(callback)
        
        # Run monitoring for a short time
        monitor_task = asyncio.create_task(self.monitor.monitor_memory())
        await asyncio.sleep(0.1)  # Let it run briefly
        monitor_task.cancel()
        
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        
        # Callback should have been called
        assert callback.called


class TestOfflineSTTService:
    """Test OfflineSTTService speech-to-text functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OfflineConfig(models_dir=self.temp_dir)
        self.model_manager = ModelManager(self.config)
        self.stt_service = OfflineSTTService(self.config, self.model_manager)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test OfflineSTTService initialization"""
        assert self.stt_service.config == self.config
        assert self.stt_service.model_manager == self.model_manager
        assert self.stt_service.model_loaded is False
        assert self.stt_service.model_path is None
    
    @patch('subprocess.run')
    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_run):
        """Test successful STT service initialization"""
        # Mock model availability
        with patch.object(self.model_manager, 'ensure_models_available', return_value={"whisper_tiny_en": True}):
            with patch.object(self.model_manager, 'get_model_path', return_value="/fake/path/model.bin"):
                # Mock whisper binary check
                mock_run.return_value = Mock(returncode=0)
                
                result = await self.stt_service.initialize()
                
                assert result is True
                assert self.stt_service.model_loaded is True
                assert self.stt_service.model_path == "/fake/path/model.bin"
    
    @patch('subprocess.run')
    @pytest.mark.asyncio
    async def test_initialize_model_unavailable(self, mock_run):
        """Test STT service initialization with unavailable model"""
        # Mock model unavailability
        with patch.object(self.model_manager, 'ensure_models_available', return_value={"whisper_tiny_en": False}):
            result = await self.stt_service.initialize()
            
            assert result is False
            assert self.stt_service.model_loaded is False
    
    @patch('subprocess.run')
    @pytest.mark.asyncio
    async def test_initialize_binary_missing(self, mock_run):
        """Test STT service initialization with missing binary"""
        # Mock model availability
        with patch.object(self.model_manager, 'ensure_models_available', return_value={"whisper_tiny_en": True}):
            with patch.object(self.model_manager, 'get_model_path', return_value="/fake/path/model.bin"):
                # Mock whisper binary missing
                mock_run.side_effect = FileNotFoundError()
                
                result = await self.stt_service.initialize()
                
                assert result is False
                assert self.stt_service.model_loaded is False
    
    @patch('subprocess.run')
    @patch('tempfile.NamedTemporaryFile')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('os.unlink')
    @pytest.mark.asyncio
    async def test_transcribe_audio_success(self, mock_unlink, mock_open, mock_exists, mock_tempfile, mock_run):
        """Test successful audio transcription"""
        # Setup mocks
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.wav"
        mock_tempfile.return_value.__enter__.return_value = mock_temp
        
        mock_run.return_value = Mock(returncode=0)
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value.read.return_value = "Hello world"
        
        # Set up service as initialized
        self.stt_service.model_loaded = True
        self.stt_service.model_path = "/fake/path/model.bin"
        
        # Test transcription
        audio_data = b"fake audio data"
        result = await self.stt_service.transcribe_audio(audio_data)
        
        assert result == "Hello world"
        assert mock_run.called
        assert mock_unlink.called
    
    @pytest.mark.asyncio
    async def test_transcribe_audio_not_initialized(self):
        """Test audio transcription when service not initialized"""
        audio_data = b"fake audio data"
        result = await self.stt_service.transcribe_audio(audio_data)
        
        assert result is None
    
    def test_cleanup(self):
        """Test STT service cleanup"""
        self.stt_service.model_loaded = True
        self.stt_service.whisper_process = Mock()
        
        self.stt_service.cleanup()
        
        assert self.stt_service.model_loaded is False
        assert self.stt_service.whisper_process.terminate.called


class TestOfflineSystem:
    """Test OfflineSystem main coordinator"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = OfflineConfig()
        self.system = OfflineSystem(self.config)
    
    def test_initialization(self):
        """Test OfflineSystem initialization"""
        assert self.system.config == self.config
        assert isinstance(self.system.network_detector, NetworkDetector)
        assert isinstance(self.system.model_manager, ModelManager)
        assert isinstance(self.system.memory_monitor, MemoryMonitor)
        assert self.system.is_offline_mode is False
        assert self.system.initialization_complete is False
    
    @patch.object(OfflineSTTService, 'initialize')
    @patch.object(OfflineTTSService, 'initialize')
    @patch.object(OfflineLLMService, 'initialize')
    @pytest.mark.asyncio
    async def test_initialize_all_services_success(self, mock_llm_init, mock_tts_init, mock_stt_init):
        """Test successful initialization of all services"""
        mock_stt_init.return_value = True
        mock_tts_init.return_value = True
        mock_llm_init.return_value = True
        
        result = await self.system.initialize()
        
        assert result is True
        assert self.system.initialization_complete is True
        assert self.system.services_ready['stt'] is True
        assert self.system.services_ready['tts'] is True
        assert self.system.services_ready['llm'] is True
    
    @patch.object(OfflineSTTService, 'initialize')
    @patch.object(OfflineTTSService, 'initialize')
    @patch.object(OfflineLLMService, 'initialize')
    @pytest.mark.asyncio
    async def test_initialize_partial_services_success(self, mock_llm_init, mock_tts_init, mock_stt_init):
        """Test initialization with some services failing"""
        mock_stt_init.return_value = True
        mock_tts_init.return_value = False  # TTS fails
        mock_llm_init.return_value = True
        
        result = await self.system.initialize()
        
        assert result is True  # System still initializes if at least one service works
        assert self.system.initialization_complete is True
        assert self.system.services_ready['stt'] is True
        assert self.system.services_ready['tts'] is False
        assert self.system.services_ready['llm'] is True
    
    @patch.object(OfflineSTTService, 'initialize')
    @patch.object(OfflineTTSService, 'initialize')
    @patch.object(OfflineLLMService, 'initialize')
    @pytest.mark.asyncio
    async def test_initialize_all_services_fail(self, mock_llm_init, mock_tts_init, mock_stt_init):
        """Test initialization failure when all services fail"""
        mock_stt_init.return_value = False
        mock_tts_init.return_value = False
        mock_llm_init.return_value = False
        
        result = await self.system.initialize()
        
        assert result is False
        assert self.system.initialization_complete is False
    
    def test_is_ready_for_offline(self):
        """Test offline readiness check"""
        # Not ready initially
        assert self.system.is_ready_for_offline() is False
        
        # Ready when all services are ready
        self.system.initialization_complete = True
        self.system.services_ready = {'stt': True, 'tts': True, 'llm': True}
        
        assert self.system.is_ready_for_offline() is True
    
    @patch.object(NetworkDetector, 'check_connectivity')
    @pytest.mark.asyncio
    async def test_check_and_switch_mode_to_offline(self, mock_check):
        """Test switching to offline mode"""
        mock_check.return_value = ConnectivityResult(
            status=ConnectivityStatus.OFFLINE,
            latency_ms=0.0,
            timestamp=time.time()
        )
        
        result = await self.system.check_and_switch_mode()
        
        assert result == ConnectivityStatus.OFFLINE
        assert self.system.is_offline_mode is True
        assert self.system.stats['mode_switches'] == 1
    
    @patch.object(NetworkDetector, 'check_connectivity')
    @pytest.mark.asyncio
    async def test_check_and_switch_mode_to_online(self, mock_check):
        """Test switching to online mode"""
        # Start in offline mode
        self.system.is_offline_mode = True
        
        mock_check.return_value = ConnectivityResult(
            status=ConnectivityStatus.ONLINE,
            latency_ms=10.0,
            timestamp=time.time()
        )
        
        result = await self.system.check_and_switch_mode()
        
        assert result == ConnectivityStatus.ONLINE
        assert self.system.is_offline_mode is False
        assert self.system.stats['mode_switches'] == 1
    
    def test_get_system_status(self):
        """Test getting system status"""
        status = self.system.get_system_status()
        
        assert isinstance(status, dict)
        assert "offline_mode" in status
        assert "initialization_complete" in status
        assert "services_ready" in status
        assert "ready_for_offline" in status
        assert "connectivity" in status
        assert "memory" in status
        assert "models" in status
        assert "statistics" in status
    
    def test_get_stats(self):
        """Test getting system statistics"""
        stats = self.system.get_stats()
        
        assert isinstance(stats, dict)
        assert "performance" in stats
        assert "services" in stats
        assert "memory" in stats
        assert "target_latency" in stats
        assert "models_status" in stats
    
    def test_reset_stats(self):
        """Test resetting statistics"""
        # Set some stats
        self.system.stats['total_offline_requests'] = 10
        self.system.stats['mode_switches'] = 5
        
        self.system.reset_stats()
        
        assert self.system.stats['total_offline_requests'] == 0
        assert self.system.stats['mode_switches'] == 0
    
    def test_reset_llm_context(self):
        """Test resetting LLM context"""
        # Mock LLM service
        mock_llm = Mock()
        self.system.offline_llm = mock_llm
        
        self.system.reset_llm_context()
        
        assert mock_llm.reset_context.called
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test system cleanup"""
        # Mock services
        mock_stt = Mock()
        mock_tts = Mock()
        mock_llm = Mock()
        
        self.system.offline_stt = mock_stt
        self.system.offline_tts = mock_tts
        self.system.offline_llm = mock_llm
        
        await self.system.cleanup()
        
        assert mock_stt.cleanup.called
        assert mock_tts.cleanup.called
        assert mock_llm.cleanup.called


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_offline_system_default(self):
        """Test creating offline system with default config"""
        system = create_offline_system()
        
        assert isinstance(system, OfflineSystem)
        assert isinstance(system.config, OfflineConfig)
    
    def test_create_offline_system_custom_config(self):
        """Test creating offline system with custom config"""
        config = OfflineConfig(target_offline_latency=2.0)
        system = create_offline_system(config)
        
        assert isinstance(system, OfflineSystem)
        assert system.config.target_offline_latency == 2.0


class TestIntegration:
    """Integration tests for offline system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = OfflineConfig(models_dir=self.temp_dir)
        self.system = create_offline_system(self.config)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_offline_system_imports(self):
        """Test that all offline system components can be imported"""
        # This test verifies that all components are properly integrated
        assert self.system is not None
        assert self.system.network_detector is not None
        assert self.system.model_manager is not None
        assert self.system.memory_monitor is not None
    
    def test_offline_config_consistency(self):
        """Test that configuration is consistent across components"""
        assert self.system.config == self.config
        assert self.system.network_detector.config == self.config
        assert self.system.model_manager.config == self.config
        assert self.system.memory_monitor.config == self.config
    
    @patch('subprocess.run')
    @patch.object(ModelManager, 'ensure_models_available')
    @patch.object(ModelManager, 'get_model_path')
    @pytest.mark.asyncio
    async def test_offline_initialization_flow(self, mock_get_path, mock_ensure, mock_run):
        """Test complete offline system initialization flow"""
        # Mock successful model availability
        mock_ensure.return_value = {
            "whisper_tiny_en": True,
            "piper_lessac_medium": True,
            "gemma_2b_q4": True
        }
        mock_get_path.return_value = "/fake/path/model.bin"
        
        # Mock successful binary checks
        mock_run.return_value = Mock(returncode=0)
        
        # Test initialization
        result = await self.system.initialize()
        
        assert result is True
        assert self.system.initialization_complete is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 