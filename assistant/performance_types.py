#!/usr/bin/env python3
"""
Performance Testing Shared Types and Data Classes
Shared types to avoid circular imports between performance and accuracy testing modules
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class PerformanceTestConfig:
    """Configuration for performance testing suite"""
    # Latency thresholds (milliseconds)
    latency_targets: Dict[str, float] = field(default_factory=lambda: {
        'cloud_p95': 800.0,
        'offline_p95': 1500.0,
        'stt_p95': 200.0,
        'llm_p95': 400.0,
        'tts_p95': 100.0
    })
    
    # Accuracy thresholds
    accuracy_targets: Dict[str, float] = field(default_factory=lambda: {
        'stt_accuracy': 0.95,
        'memory_recall_bleu': 0.8,
        'memory_recall_semantic': 0.85,
        'ocr_accuracy': 0.92
    })
    
    # Stress testing parameters
    stress_test_config: Dict[str, Any] = field(default_factory=lambda: {
        'max_concurrent_users': 100,
        'test_duration_seconds': 300,
        'ramp_up_duration_seconds': 60,
        'memory_leak_threshold_percent': 10
    })
    
    # Regression detection settings
    regression_config: Dict[str, Any] = field(default_factory=lambda: {
        'anomaly_contamination': 0.1,
        'z_score_threshold': 2.5,
        'minimum_samples': 50
    })


@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    status: str  # 'passed', 'failed', 'warning'
    metrics: Dict[str, float]
    execution_time: float
    timestamp: datetime
    details: Optional[str] = None
    errors: List[str] = field(default_factory=list)


@dataclass
class PerformanceTestReport:
    """Comprehensive performance test report"""
    test_session_id: str
    timestamp: datetime
    overall_status: str  # 'passed', 'failed', 'partial'
    test_results: List[TestResult]
    summary_metrics: Dict[str, float]
    recommendations: List[str]
    execution_time: float 