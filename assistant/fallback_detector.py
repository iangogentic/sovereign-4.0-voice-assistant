"""
Sovereign 4.0 Voice Assistant - Mode Validation and Error Handling System

Provides comprehensive validation and error handling for operation modes:
- API availability checking for REALTIME_ONLY mode
- Traditional pipeline verification for TRADITIONAL_ONLY mode
- Mode-specific configuration validation with detailed error messages
- User-friendly warning system for mode limitations
- Graceful degradation strategies for mode failures

This system works closely with ModeManager to ensure proper mode operation
and provides intelligent fallback mechanisms when issues are detected.

Usage:
    validator = ModeValidator(config, mode_manager)
    
    # Check if a mode can be used
    is_valid, issues = await validator.validate_mode(OperationMode.REALTIME_ONLY)
    
    # Check API availability
    api_status = await validator.check_api_availability()
    
    # Handle mode failure with graceful degradation
    await validator.handle_mode_failure(OperationMode.REALTIME_ONLY, error)
"""

import asyncio
import logging
import time
import aiohttp
import traceback
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from config_manager import SovereignConfig, OperationMode
from mode_switch_manager import ModeManager, ModeValidationError, ModeTransitionError

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class FailureType(Enum):
    """Types of mode failures"""
    API_UNAVAILABLE = "api_unavailable"
    AUTHENTICATION_FAILED = "authentication_failed"
    NETWORK_ERROR = "network_error"
    SERVICE_TIMEOUT = "service_timeout"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ValidationIssue:
    """Represents a validation issue found during mode checking"""
    code: str
    severity: ValidationSeverity
    message: str
    details: Optional[str] = None
    suggestion: Optional[str] = None
    component: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class APIStatus:
    """Status information for an API endpoint"""
    service_name: str
    is_available: bool
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    last_checked: datetime = field(default_factory=datetime.now)
    status_code: Optional[int] = None

@dataclass
class ModeHealthStatus:
    """Comprehensive health status for an operation mode"""
    mode: OperationMode
    is_healthy: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    api_statuses: List[APIStatus] = field(default_factory=list)
    last_validated: datetime = field(default_factory=datetime.now)
    estimated_reliability: float = 1.0  # 0.0 to 1.0

@dataclass
class DegradationStrategy:
    """Strategy for graceful degradation when a mode fails"""
    from_mode: OperationMode
    fallback_mode: Optional[OperationMode]
    reason: str
    user_message: str
    automatic: bool = True
    cooldown_seconds: int = 60

class ModeValidator:
    """
    Comprehensive validation and error handling system for operation modes
    
    Provides mode validation, API health checking, and graceful degradation
    strategies to ensure reliable operation across all modes.
    """
    
    def __init__(self, config: SovereignConfig, mode_manager: ModeManager):
        """Initialize the mode validator"""
        self.config = config
        self.mode_manager = mode_manager
        self.logger = logger
        
        # Validation state
        self._health_cache: Dict[OperationMode, ModeHealthStatus] = {}
        self._api_cache: Dict[str, APIStatus] = {}
        self._cache_duration = timedelta(minutes=5)
        
        # Degradation tracking
        self._failure_counts: Dict[OperationMode, int] = {}
        self._last_failures: Dict[OperationMode, datetime] = {}
        self._cooldown_until: Dict[OperationMode, datetime] = {}
        
        # Configuration
        self.validation_timeout = 10.0  # seconds
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        self.failure_threshold = 3  # failures before degradation
        
        # Callbacks
        self.on_validation_issue: Optional[Callable[[ValidationIssue], None]] = None
        self.on_mode_degraded: Optional[Callable[[DegradationStrategy], None]] = None
        self.on_mode_recovered: Optional[Callable[[OperationMode], None]] = None
        
        logger.info("ModeValidator initialized")
    
    async def validate_mode(self, mode: OperationMode, force_refresh: bool = False) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate if a mode can be used effectively
        
        Args:
            mode: Operation mode to validate
            force_refresh: Force fresh validation, ignore cache
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        # Check cache first unless force refresh
        if not force_refresh and mode in self._health_cache:
            cached_status = self._health_cache[mode]
            if datetime.now() - cached_status.last_validated < self._cache_duration:
                return cached_status.is_healthy, cached_status.issues
        
        logger.info(f"Validating mode: {mode.value}")
        issues = []
        
        try:
            # Mode-specific validation
            if mode == OperationMode.REALTIME_ONLY:
                issues.extend(await self._validate_realtime_mode())
            elif mode == OperationMode.TRADITIONAL_ONLY:
                issues.extend(await self._validate_traditional_mode())
            elif mode == OperationMode.HYBRID_AUTO:
                issues.extend(await self._validate_hybrid_mode())
            
            # Common validation for all modes
            issues.extend(await self._validate_common_requirements(mode))
            
            # Determine overall health
            is_healthy = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] 
                               for issue in issues)
            
            # Calculate estimated reliability
            error_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
            warning_count = sum(1 for issue in issues if issue.severity == ValidationSeverity.WARNING)
            reliability = max(0.0, 1.0 - (error_count * 0.3) - (warning_count * 0.1))
            
            # Cache the result
            health_status = ModeHealthStatus(
                mode=mode,
                is_healthy=is_healthy,
                issues=issues,
                estimated_reliability=reliability
            )
            self._health_cache[mode] = health_status
            
            # Notify of issues
            for issue in issues:
                if self.on_validation_issue:
                    self.on_validation_issue(issue)
            
            logger.info(f"Mode {mode.value} validation completed: healthy={is_healthy}, "
                       f"issues={len(issues)}, reliability={reliability:.2f}")
            
            return is_healthy, issues
            
        except Exception as e:
            critical_issue = ValidationIssue(
                code="VALIDATION_FAILED",
                severity=ValidationSeverity.CRITICAL,
                message=f"Mode validation failed: {str(e)}",
                details=traceback.format_exc(),
                component="mode_validator"
            )
            logger.error(f"Mode validation failed for {mode.value}: {e}")
            return False, [critical_issue]
    
    async def _validate_realtime_mode(self) -> List[ValidationIssue]:
        """Validate REALTIME_ONLY mode requirements"""
        issues = []
        
        # Check OpenAI API key
        if not self.config.development.mock_apis:
            if not self.config.api.openai_api_key:
                issues.append(ValidationIssue(
                    code="MISSING_OPENAI_KEY",
                    severity=ValidationSeverity.ERROR,
                    message="OpenAI API key is required for Realtime mode",
                    suggestion="Set OPENAI_API_KEY environment variable or configure in settings",
                    component="api_config"
                ))
        
        # Check Realtime API configuration
        if not self.config.realtime_api.enabled:
            issues.append(ValidationIssue(
                code="REALTIME_API_DISABLED",
                severity=ValidationSeverity.ERROR,
                message="Realtime API is not enabled in configuration",
                suggestion="Set realtime_api.enabled to true in configuration",
                component="realtime_config"
            ))
        
        # Check API availability
        api_status = await self.check_openai_api_availability()
        if not api_status.is_available:
            issues.append(ValidationIssue(
                code="OPENAI_API_UNAVAILABLE",
                severity=ValidationSeverity.ERROR,
                message=f"OpenAI API is not available: {api_status.error_message}",
                suggestion="Check internet connection and API status",
                component="openai_api"
            ))
        elif api_status.response_time_ms and api_status.response_time_ms > 5000:
            issues.append(ValidationIssue(
                code="SLOW_API_RESPONSE",
                severity=ValidationSeverity.WARNING,
                message=f"OpenAI API response time is slow: {api_status.response_time_ms:.0f}ms",
                suggestion="Consider checking network connection",
                component="openai_api"
            ))
        
        return issues
    
    async def _validate_traditional_mode(self) -> List[ValidationIssue]:
        """Validate TRADITIONAL_ONLY mode requirements"""
        issues = []
        
        # Check STT provider configuration
        if not self.config.stt.primary_provider:
            issues.append(ValidationIssue(
                code="MISSING_STT_PROVIDER",
                severity=ValidationSeverity.ERROR,
                message="STT provider is not configured",
                suggestion="Configure stt.primary_provider in settings",
                component="stt_config"
            ))
        
        # Check TTS provider configuration
        if not self.config.tts.primary_provider:
            issues.append(ValidationIssue(
                code="MISSING_TTS_PROVIDER",
                severity=ValidationSeverity.ERROR,
                message="TTS provider is not configured",
                suggestion="Configure tts.primary_provider in settings",
                component="tts_config"
            ))
        
        # Check if traditional pipeline services are available
        pipeline_status = await self.check_traditional_pipeline_availability()
        if not pipeline_status["stt_available"]:
            issues.append(ValidationIssue(
                code="STT_SERVICE_UNAVAILABLE",
                severity=ValidationSeverity.ERROR,
                message="STT service is not available",
                suggestion="Check STT service configuration and connectivity",
                component="stt_service"
            ))
        
        if not pipeline_status["tts_available"]:
            issues.append(ValidationIssue(
                code="TTS_SERVICE_UNAVAILABLE",
                severity=ValidationSeverity.ERROR,
                message="TTS service is not available",
                suggestion="Check TTS service configuration and connectivity",
                component="tts_service"
            ))
        
        # Warn if Realtime API is enabled but mode is traditional
        if self.config.realtime_api.enabled:
            issues.append(ValidationIssue(
                code="REALTIME_API_IGNORED",
                severity=ValidationSeverity.WARNING,
                message="Realtime API is enabled but will be ignored in TRADITIONAL_ONLY mode",
                suggestion="Consider switching to HYBRID_AUTO mode to use both",
                component="mode_config"
            ))
        
        return issues
    
    async def _validate_hybrid_mode(self) -> List[ValidationIssue]:
        """Validate HYBRID_AUTO mode requirements"""
        issues = []
        
        # Hybrid mode needs both realtime and traditional capabilities as fallbacks
        realtime_issues = await self._validate_realtime_mode()
        traditional_issues = await self._validate_traditional_mode()
        
        # Filter out the "API ignored" warning since it doesn't apply to hybrid mode
        traditional_issues = [
            issue for issue in traditional_issues 
            if issue.code != "REALTIME_API_IGNORED"
        ]
        
        # For hybrid mode, realtime issues are warnings (not errors) since we can fall back
        for issue in realtime_issues:
            if issue.severity == ValidationSeverity.ERROR:
                issues.append(ValidationIssue(
                    code=f"HYBRID_{issue.code}",
                    severity=ValidationSeverity.WARNING,
                    message=f"Realtime capability limited: {issue.message}",
                    suggestion=f"Realtime features will be unavailable. {issue.suggestion}",
                    component=issue.component
                ))
            else:
                issues.append(issue)
        
        # Traditional issues remain errors since we need fallback capability
        for issue in traditional_issues:
            if issue.severity == ValidationSeverity.ERROR:
                issues.append(ValidationIssue(
                    code=f"HYBRID_{issue.code}",
                    severity=ValidationSeverity.ERROR,
                    message=f"Fallback capability missing: {issue.message}",
                    suggestion=f"Required for fallback functionality. {issue.suggestion}",
                    component=issue.component
                ))
            else:
                issues.append(issue)
        
        return issues
    
    async def _validate_common_requirements(self, mode: OperationMode) -> List[ValidationIssue]:
        """Validate common requirements for all modes"""
        issues = []
        
        # Check if mode manager is properly initialized
        if not self.mode_manager.is_initialized():
            issues.append(ValidationIssue(
                code="MODE_MANAGER_NOT_INITIALIZED",
                severity=ValidationSeverity.CRITICAL,
                message="Mode manager is not initialized",
                suggestion="Initialize mode manager before using",
                component="mode_manager"
            ))
        
        # Check for mode capability compatibility
        try:
            capabilities = self.mode_manager.get_capabilities(mode)
            
            # Verify capabilities match mode expectations
            if mode == OperationMode.REALTIME_ONLY and not capabilities.can_use_realtime_api:
                issues.append(ValidationIssue(
                    code="CAPABILITY_MISMATCH",
                    severity=ValidationSeverity.ERROR,
                    message="Mode capabilities don't match expected realtime functionality",
                    component="mode_capabilities"
                ))
                
        except Exception as e:
            issues.append(ValidationIssue(
                code="CAPABILITY_CHECK_FAILED",
                severity=ValidationSeverity.ERROR,
                message=f"Failed to check mode capabilities: {str(e)}",
                component="mode_capabilities"
            ))
        
        return issues
    
    async def check_openai_api_availability(self) -> APIStatus:
        """Check if OpenAI API is available and responsive"""
        service_name = "openai_api"
        
        # Check cache first
        if service_name in self._api_cache:
            cached_status = self._api_cache[service_name]
            if datetime.now() - cached_status.last_checked < timedelta(minutes=2):
                return cached_status
        
        start_time = time.time()
        
        try:
            # Use mock response if in development mode
            if self.config.development.mock_apis:
                await asyncio.sleep(0.1)  # Simulate network delay
                status = APIStatus(
                    service_name=service_name,
                    is_available=True,
                    response_time_ms=100,
                    status_code=200
                )
            else:
                # Make a simple request to OpenAI API
                headers = {
                    "Authorization": f"Bearer {self.config.api.openai_api_key}",
                    "Content-Type": "application/json"
                }
                
                timeout = aiohttp.ClientTimeout(total=self.validation_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(
                        f"{self.config.api.openai_base_url}/models",
                        headers=headers
                    ) as response:
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status == 200:
                            status = APIStatus(
                                service_name=service_name,
                                is_available=True,
                                response_time_ms=response_time,
                                status_code=response.status
                            )
                        else:
                            status = APIStatus(
                                service_name=service_name,
                                is_available=False,
                                response_time_ms=response_time,
                                error_message=f"HTTP {response.status}: {await response.text()}",
                                status_code=response.status
                            )
                            
        except asyncio.TimeoutError:
            status = APIStatus(
                service_name=service_name,
                is_available=False,
                error_message="Request timeout"
            )
        except Exception as e:
            status = APIStatus(
                service_name=service_name,
                is_available=False,
                error_message=str(e)
            )
        
        # Cache the result
        self._api_cache[service_name] = status
        
        logger.debug(f"OpenAI API status: available={status.is_available}, "
                    f"response_time={status.response_time_ms}ms")
        
        return status
    
    async def check_traditional_pipeline_availability(self) -> Dict[str, bool]:
        """Check if traditional pipeline components are available"""
        
        # For now, return mock status - in a real implementation,
        # this would check actual STT/TTS service availability
        if self.config.development.mock_apis:
            return {
                "stt_available": True,
                "tts_available": True,
                "llm_available": True
            }
        
        # TODO: Implement actual service checks
        # This would check if Whisper, TTS, and LLM services are responsive
        return {
            "stt_available": bool(self.config.stt.primary_provider),
            "tts_available": bool(self.config.tts.primary_provider),
            "llm_available": True  # Assume LLM router is always available
        }
    
    async def handle_mode_failure(self, failed_mode: OperationMode, error: Exception, failure_type: FailureType = FailureType.UNKNOWN_ERROR) -> bool:
        """
        Handle a mode failure with graceful degradation
        
        Args:
            failed_mode: Mode that failed
            error: Exception that caused the failure
            failure_type: Type of failure for better handling
            
        Returns:
            True if degradation was successful, False otherwise
        """
        logger.warning(f"Handling failure in mode {failed_mode.value}: {error}")
        
        # Update failure tracking
        self._failure_counts[failed_mode] = self._failure_counts.get(failed_mode, 0) + 1
        self._last_failures[failed_mode] = datetime.now()
        
        # Determine degradation strategy
        strategy = self._get_degradation_strategy(failed_mode, failure_type)
        
        if strategy.fallback_mode:
            try:
                # Attempt to switch to fallback mode
                success = await self.mode_manager.switch_mode(
                    strategy.fallback_mode, 
                    f"degradation: {strategy.reason}"
                )
                
                if success:
                    # Set cooldown period for failed mode
                    self._cooldown_until[failed_mode] = (
                        datetime.now() + timedelta(seconds=strategy.cooldown_seconds)
                    )
                    
                    logger.info(f"Successfully degraded from {failed_mode.value} to "
                               f"{strategy.fallback_mode.value}")
                    
                    # Notify about degradation
                    if self.on_mode_degraded:
                        self.on_mode_degraded(strategy)
                    
                    return True
                else:
                    logger.error(f"Failed to switch to fallback mode {strategy.fallback_mode.value}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during mode degradation: {e}")
                return False
        else:
            logger.error(f"No fallback available for failed mode {failed_mode.value}")
            return False
    
    def _get_degradation_strategy(self, failed_mode: OperationMode, failure_type: FailureType) -> DegradationStrategy:
        """Determine the best degradation strategy for a failed mode"""
        
        if failed_mode == OperationMode.REALTIME_ONLY:
            return DegradationStrategy(
                from_mode=failed_mode,
                fallback_mode=OperationMode.TRADITIONAL_ONLY,
                reason=f"Realtime API failure: {failure_type.value}",
                user_message="Switched to traditional voice processing due to connectivity issues",
                cooldown_seconds=120
            )
        elif failed_mode == OperationMode.HYBRID_AUTO:
            # For hybrid mode, try to fall back to traditional
            return DegradationStrategy(
                from_mode=failed_mode,
                fallback_mode=OperationMode.TRADITIONAL_ONLY,
                reason=f"Hybrid mode failure: {failure_type.value}",
                user_message="Temporarily using traditional voice processing",
                cooldown_seconds=60
            )
        else:  # TRADITIONAL_ONLY
            # No good fallback for traditional mode failure
            return DegradationStrategy(
                from_mode=failed_mode,
                fallback_mode=None,
                reason=f"Traditional pipeline failure: {failure_type.value}",
                user_message="Voice assistant is temporarily unavailable",
                automatic=False
            )
    
    async def check_recovery_opportunity(self, failed_mode: OperationMode) -> bool:
        """Check if a failed mode can be recovered"""
        
        # Check if still in cooldown
        if failed_mode in self._cooldown_until:
            if datetime.now() < self._cooldown_until[failed_mode]:
                return False
            else:
                # Cooldown expired, remove it
                del self._cooldown_until[failed_mode]
        
        # Validate the mode to see if it's healthy now
        is_healthy, issues = await self.validate_mode(failed_mode, force_refresh=True)
        
        if is_healthy:
            # Reset failure count on successful recovery
            self._failure_counts[failed_mode] = 0
            
            logger.info(f"Recovery opportunity detected for mode {failed_mode.value}")
            
            if self.on_mode_recovered:
                self.on_mode_recovered(failed_mode)
            
            return True
        
        return False
    
    def get_mode_health_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of health status for all modes"""
        summary = {}
        
        for mode in OperationMode:
            mode_data = {
                "mode": mode.value,
                "failure_count": self._failure_counts.get(mode, 0),
                "last_failure": (
                    self._last_failures[mode].isoformat() 
                    if mode in self._last_failures else None
                ),
                "in_cooldown": (
                    mode in self._cooldown_until and 
                    datetime.now() < self._cooldown_until[mode]
                ),
                "cooldown_until": (
                    self._cooldown_until[mode].isoformat() 
                    if mode in self._cooldown_until else None
                )
            }
            
            # Add cached health info if available
            if mode in self._health_cache:
                health = self._health_cache[mode]
                mode_data.update({
                    "is_healthy": health.is_healthy,
                    "reliability": health.estimated_reliability,
                    "issue_count": len(health.issues),
                    "last_validated": health.last_validated.isoformat()
                })
            
            summary[mode.value] = mode_data
        
        return summary
    
    def clear_cache(self) -> None:
        """Clear all cached validation and API status data"""
        self._health_cache.clear()
        self._api_cache.clear()
        logger.info("Validation cache cleared")

# Convenience functions
def create_mode_validator(config: SovereignConfig, mode_manager: ModeManager) -> ModeValidator:
    """Factory function to create a ModeValidator instance"""
    return ModeValidator(config, mode_manager) 