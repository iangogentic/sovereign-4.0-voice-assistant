"""
Predictive Error Enrichment System for Sovereign 4.0 Voice Assistant

Enhances error handling with predictive analytics context including root cause analysis,
similar incident patterns, and recovery recommendations following 2024-2025 best practices.

Key Features:
- Asynchronous error enrichment to avoid performance impact
- Predictive root cause analysis
- Similar incident pattern matching
- Recovery recommendation generation
- Error correlation and trend analysis
- Graceful degradation when predictive services unavailable
"""

import asyncio
import logging
import time
import threading
import json
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque, defaultdict
import hashlib

from .error_handling import (
    VoiceAIException, ErrorCategory, ErrorContext, 
    STTException, LLMException, TTSException, AudioException
)
from .predictive_analytics import PredictiveAnalyticsFramework
from .structured_logging import VoiceAILogger, get_voice_ai_logger
from .metrics_collector import MetricsCollector
from .drift_detection import DriftDetector

@dataclass
class ErrorPattern:
    """Pattern identified from historical errors"""
    pattern_id: str
    error_type: str
    common_context: Dict[str, Any]
    frequency: int
    last_occurrence: float
    success_rate_after: float
    typical_resolution_time: float
    recovery_strategies: List[str]

@dataclass
class SimilarIncident:
    """Similar incident found in error history"""
    incident_id: str
    error_message: str
    error_category: str
    context_similarity: float
    time_difference: float
    resolution_method: Optional[str]
    resolution_success: bool
    lessons_learned: List[str]

@dataclass
class PredictiveRootCause:
    """Predicted root cause with confidence scoring"""
    cause_type: str
    description: str
    confidence: float  # 0.0 to 1.0
    contributing_factors: List[str]
    detection_method: str  # drift, pattern, anomaly, etc.
    time_to_impact: Optional[float]  # seconds before error occurred
    preventive_measures: List[str]

@dataclass
class RecoveryRecommendation:
    """Recommended recovery action with metadata"""
    action_type: str
    description: str
    priority: str  # low, medium, high, urgent
    success_probability: float  # 0.0 to 1.0
    estimated_time: Optional[float]  # seconds
    automation_possible: bool
    dependencies: List[str]
    risk_level: str  # low, medium, high
    rollback_plan: Optional[str] = None

@dataclass
class EnrichedErrorContext:
    """Enhanced error context with predictive insights"""
    original_error: VoiceAIException
    enrichment_timestamp: float
    enrichment_duration_ms: float
    
    # Predictive insights
    predicted_root_causes: List[PredictiveRootCause]
    similar_incidents: List[SimilarIncident]
    recovery_recommendations: List[RecoveryRecommendation]
    error_patterns: List[ErrorPattern]
    
    # Context analysis
    system_state_at_error: Dict[str, Any]
    leading_indicators: List[Dict[str, Any]]
    degradation_timeline: List[Dict[str, Any]]
    
    # Correlation analysis
    correlated_errors: List[str]
    error_cluster_id: Optional[str]
    cascade_probability: float
    
    # Impact assessment
    severity_escalation_risk: float
    user_impact_score: float
    business_impact_assessment: str
    
    # Confidence metrics
    enrichment_confidence: float
    prediction_reliability: float
    data_quality_score: float

class PredictiveErrorEnricher:
    """
    Predictive Error Enricher for enhanced error handling
    
    Provides comprehensive error enrichment by combining:
    - Historical error pattern analysis
    - Predictive analytics insights
    - System state correlation
    - Recovery recommendation generation
    """
    
    def __init__(
        self,
        predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        drift_detector: Optional[DriftDetector] = None,
        logger: Optional[VoiceAILogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.predictive_analytics = predictive_analytics
        self.metrics_collector = metrics_collector
        self.drift_detector = drift_detector
        self.logger = logger or get_voice_ai_logger("predictive_error_enricher")
        
        # Configuration
        self.config = config or self._get_default_config()
        
        # Error history and pattern storage
        self._lock = threading.RLock()
        self.error_history: deque = deque(maxlen=self.config['max_error_history'])
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.incident_cache: Dict[str, SimilarIncident] = {}
        
        # Error clustering and correlation
        self.error_clusters: Dict[str, List[str]] = defaultdict(list)
        self.correlation_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.enrichment_stats = {
            'total_enrichments': 0,
            'successful_enrichments': 0,
            'average_enrichment_time_ms': 0.0,
            'cache_hits': 0,
            'prediction_accuracy': 0.0
        }
        
        # Async processing queue
        self.enrichment_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.enrichment_tasks: List[asyncio.Task] = []
        
        self.logger.info("🔍 PredictiveErrorEnricher initialized with predictive capabilities")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for error enrichment"""
        return {
            'max_error_history': 10000,
            'pattern_similarity_threshold': 0.8,
            'incident_similarity_threshold': 0.7,
            'max_similar_incidents': 5,
            'max_root_causes': 3,
            'max_recommendations': 5,
            'async_enrichment_enabled': True,
            'enrichment_timeout_ms': 5000,
            'cache_ttl_hours': 24,
            'min_pattern_frequency': 3,
            'correlation_window_hours': 6,
            'drift_correlation_enabled': True,
            'system_state_capture_enabled': True
        }
    
    async def enrich_error(
        self, 
        error: VoiceAIException, 
        async_mode: bool = True
    ) -> EnrichedErrorContext:
        """
        Enrich error with predictive context
        
        Args:
            error: The error to enrich
            async_mode: Whether to perform enrichment asynchronously
            
        Returns:
            EnrichedErrorContext with predictive insights
        """
        if async_mode and self.config['async_enrichment_enabled']:
            # Queue for async processing
            enrichment_future = asyncio.Future()
            await self.enrichment_queue.put((error, enrichment_future))
            return await enrichment_future
        else:
            # Synchronous enrichment
            return await self._perform_enrichment(error)
    
    async def _perform_enrichment(self, error: VoiceAIException) -> EnrichedErrorContext:
        """Perform comprehensive error enrichment"""
        start_time = time.time()
        
        try:
            # Generate error fingerprint for caching
            error_fingerprint = self._generate_error_fingerprint(error)
            
            # Check cache first
            cached_enrichment = self._get_cached_enrichment(error_fingerprint)
            if cached_enrichment:
                self.enrichment_stats['cache_hits'] += 1
                return cached_enrichment
            
            # Capture system state at error
            system_state = await self._capture_system_state()
            
            # Analyze error patterns
            error_patterns = self._analyze_error_patterns(error)
            
            # Find similar incidents
            similar_incidents = await self._find_similar_incidents(error)
            
            # Predict root causes
            predicted_root_causes = await self._predict_root_causes(error, system_state)
            
            # Generate recovery recommendations
            recovery_recommendations = await self._generate_recovery_recommendations(
                error, predicted_root_causes, similar_incidents
            )
            
            # Analyze correlations
            correlated_errors, cluster_id = self._analyze_error_correlations(error)
            
            # Calculate impact assessments
            severity_risk = self._calculate_severity_escalation_risk(error, predicted_root_causes)
            user_impact = self._calculate_user_impact_score(error, system_state)
            business_impact = self._assess_business_impact(error, severity_risk, user_impact)
            
            # Get leading indicators and timeline
            leading_indicators = await self._get_leading_indicators(error)
            degradation_timeline = await self._build_degradation_timeline(error)
            
            # Calculate confidence metrics
            enrichment_confidence = self._calculate_enrichment_confidence(
                predicted_root_causes, similar_incidents, recovery_recommendations
            )
            prediction_reliability = self._calculate_prediction_reliability()
            data_quality_score = self._calculate_data_quality_score(system_state)
            
            # Calculate cascade probability
            cascade_probability = self._calculate_cascade_probability(error, correlated_errors)
            
            # Create enriched context
            enrichment_duration_ms = (time.time() - start_time) * 1000
            
            enriched_context = EnrichedErrorContext(
                original_error=error,
                enrichment_timestamp=time.time(),
                enrichment_duration_ms=enrichment_duration_ms,
                predicted_root_causes=predicted_root_causes,
                similar_incidents=similar_incidents,
                recovery_recommendations=recovery_recommendations,
                error_patterns=error_patterns,
                system_state_at_error=system_state,
                leading_indicators=leading_indicators,
                degradation_timeline=degradation_timeline,
                correlated_errors=correlated_errors,
                error_cluster_id=cluster_id,
                cascade_probability=cascade_probability,
                severity_escalation_risk=severity_risk,
                user_impact_score=user_impact,
                business_impact_assessment=business_impact,
                enrichment_confidence=enrichment_confidence,
                prediction_reliability=prediction_reliability,
                data_quality_score=data_quality_score
            )
            
            # Cache the enrichment
            self._cache_enrichment(error_fingerprint, enriched_context)
            
            # Update statistics
            self._update_enrichment_stats(enrichment_duration_ms, True)
            
            # Store error for pattern analysis
            self._store_error_for_analysis(error, enriched_context)
            
            self.logger.info(
                f"🔍 Error enriched successfully: {error.__class__.__name__} "
                f"({enrichment_duration_ms:.1f}ms, confidence: {enrichment_confidence:.2f})"
            )
            
            return enriched_context
            
        except Exception as e:
            enrichment_duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Error enrichment failed: {e}")
            self._update_enrichment_stats(enrichment_duration_ms, False)
            
            # Return minimal enrichment
            return self._create_minimal_enrichment(error, enrichment_duration_ms)
    
    def _generate_error_fingerprint(self, error: VoiceAIException) -> str:
        """Generate unique fingerprint for error caching"""
        # Create hash from error type, message, and relevant context
        fingerprint_data = {
            'error_type': error.__class__.__name__,
            'error_category': error.category.value if hasattr(error, 'category') else 'unknown',
            'message_hash': hashlib.md5(str(error).encode()).hexdigest()[:8]
        }
        
        # Add context if available
        if hasattr(error, 'context') and error.context:
            context_keys = ['service_name', 'operation', 'model_used']
            context_data = {k: error.context.get(k) for k in context_keys if k in error.context}
            fingerprint_data.update(context_data)
        
        fingerprint_string = json.dumps(fingerprint_data, sort_keys=True)
        return hashlib.sha256(fingerprint_string.encode()).hexdigest()[:16]
    
    def _get_cached_enrichment(self, fingerprint: str) -> Optional[EnrichedErrorContext]:
        """Get cached enrichment if available and not expired"""
        # Implement simple in-memory cache (in production, use Redis or similar)
        cache_key = f"enrichment_{fingerprint}"
        
        with self._lock:
            if hasattr(self, '_enrichment_cache'):
                cached_data = self._enrichment_cache.get(cache_key)
                if cached_data:
                    enrichment, cache_time = cached_data
                    cache_age_hours = (time.time() - cache_time) / 3600
                    if cache_age_hours < self.config['cache_ttl_hours']:
                        return enrichment
        
        return None
    
    def _cache_enrichment(self, fingerprint: str, enrichment: EnrichedErrorContext):
        """Cache enrichment for future use"""
        if not hasattr(self, '_enrichment_cache'):
            self._enrichment_cache = {}
        
        cache_key = f"enrichment_{fingerprint}"
        with self._lock:
            self._enrichment_cache[cache_key] = (enrichment, time.time())
    
    async def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        system_state = {
            'timestamp': time.time(),
            'capture_method': 'real_time'
        }
        
        try:
            # Capture metrics if available
            if self.metrics_collector:
                current_metrics = self.metrics_collector.get_current_metrics()
                system_state['metrics'] = current_metrics
            
            # Capture drift status if available
            if self.drift_detector:
                drift_status = await self.drift_detector.get_current_status()
                system_state['drift_status'] = drift_status
            
            # Capture predictive analytics state if available
            if self.predictive_analytics:
                analytics_state = await self.predictive_analytics.get_current_state()
                system_state['analytics_state'] = analytics_state
            
            # Basic system information
            system_state['system_info'] = {
                'timestamp': time.time(),
                'active_components': self._get_active_components()
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to capture complete system state: {e}")
            system_state['capture_error'] = str(e)
        
        return system_state
    
    def _analyze_error_patterns(self, error: VoiceAIException) -> List[ErrorPattern]:
        """Analyze historical error patterns similar to current error"""
        matching_patterns = []
        
        try:
            error_type = error.__class__.__name__
            error_category = error.category.value if hasattr(error, 'category') else 'unknown'
            
            with self._lock:
                for pattern_id, pattern in self.error_patterns.items():
                    # Check if pattern matches current error
                    if (pattern.error_type == error_type or 
                        pattern.common_context.get('category') == error_category):
                        
                        # Calculate similarity score
                        similarity = self._calculate_pattern_similarity(error, pattern)
                        if similarity >= self.config['pattern_similarity_threshold']:
                            matching_patterns.append(pattern)
            
            # Sort by frequency and recency
            matching_patterns.sort(
                key=lambda p: (p.frequency, -abs(time.time() - p.last_occurrence)),
                reverse=True
            )
            
        except Exception as e:
            self.logger.warning(f"Error analyzing patterns: {e}")
        
        return matching_patterns[:5]  # Return top 5 patterns
    
    async def _find_similar_incidents(self, error: VoiceAIException) -> List[SimilarIncident]:
        """Find similar incidents from error history"""
        similar_incidents = []
        
        try:
            error_message = str(error)
            error_category = error.category.value if hasattr(error, 'category') else 'unknown'
            
            with self._lock:
                for incident in list(self.error_history):
                    # Calculate similarity
                    similarity = self._calculate_incident_similarity(error, incident)
                    
                    if similarity >= self.config['incident_similarity_threshold']:
                        similar_incident = SimilarIncident(
                            incident_id=incident.get('id', 'unknown'),
                            error_message=incident.get('message', ''),
                            error_category=incident.get('category', 'unknown'),
                            context_similarity=similarity,
                            time_difference=time.time() - incident.get('timestamp', 0),
                            resolution_method=incident.get('resolution_method'),
                            resolution_success=incident.get('resolution_success', False),
                            lessons_learned=incident.get('lessons_learned', [])
                        )
                        similar_incidents.append(similar_incident)
            
            # Sort by similarity and recency
            similar_incidents.sort(
                key=lambda i: (i.context_similarity, -i.time_difference),
                reverse=True
            )
            
        except Exception as e:
            self.logger.warning(f"Error finding similar incidents: {e}")
        
        return similar_incidents[:self.config['max_similar_incidents']]
    
    async def _predict_root_causes(
        self, 
        error: VoiceAIException, 
        system_state: Dict[str, Any]
    ) -> List[PredictiveRootCause]:
        """Predict potential root causes using predictive analytics"""
        root_causes = []
        
        try:
            # Drift-based root cause analysis
            if self.drift_detector:
                drift_causes = await self._analyze_drift_root_causes(error)
                root_causes.extend(drift_causes)
            
            # Pattern-based root cause analysis
            pattern_causes = self._analyze_pattern_root_causes(error, system_state)
            root_causes.extend(pattern_causes)
            
            # Anomaly-based root cause analysis
            if self.predictive_analytics:
                anomaly_causes = await self._analyze_anomaly_root_causes(error, system_state)
                root_causes.extend(anomaly_causes)
            
            # Sort by confidence
            root_causes.sort(key=lambda c: c.confidence, reverse=True)
            
        except Exception as e:
            self.logger.warning(f"Error predicting root causes: {e}")
        
        return root_causes[:self.config['max_root_causes']]
    
    async def _generate_recovery_recommendations(
        self,
        error: VoiceAIException,
        root_causes: List[PredictiveRootCause],
        similar_incidents: List[SimilarIncident]
    ) -> List[RecoveryRecommendation]:
        """Generate recovery recommendations based on analysis"""
        recommendations = []
        
        try:
            # Recommendations from root causes
            for cause in root_causes:
                for measure in cause.preventive_measures:
                    recommendation = RecoveryRecommendation(
                        action_type="root_cause_mitigation",
                        description=measure,
                        priority="high" if cause.confidence > 0.8 else "medium",
                        success_probability=cause.confidence,
                        estimated_time=None,
                        automation_possible=self._is_automatable(measure),
                        dependencies=[],
                        risk_level="low" if cause.confidence > 0.7 else "medium"
                    )
                    recommendations.append(recommendation)
            
            # Recommendations from similar incidents
            for incident in similar_incidents:
                if incident.resolution_success and incident.resolution_method:
                    recommendation = RecoveryRecommendation(
                        action_type="historical_resolution",
                        description=incident.resolution_method,
                        priority="medium",
                        success_probability=0.8 if incident.resolution_success else 0.4,
                        estimated_time=None,
                        automation_possible=False,
                        dependencies=[],
                        risk_level="low"
                    )
                    recommendations.append(recommendation)
            
            # Standard recovery recommendations by error type
            standard_recommendations = self._get_standard_recovery_recommendations(error)
            recommendations.extend(standard_recommendations)
            
            # Remove duplicates and sort by priority and success probability
            unique_recommendations = self._deduplicate_recommendations(recommendations)
            unique_recommendations.sort(
                key=lambda r: (
                    {'urgent': 4, 'high': 3, 'medium': 2, 'low': 1}[r.priority],
                    r.success_probability
                ),
                reverse=True
            )
            
        except Exception as e:
            self.logger.warning(f"Error generating recovery recommendations: {e}")
        
        return unique_recommendations[:self.config['max_recommendations']]
    
    async def _analyze_drift_root_causes(self, error: VoiceAIException) -> List[PredictiveRootCause]:
        """Analyze drift-related root causes"""
        drift_causes = []
        
        try:
            if self.drift_detector:
                active_alerts = await self.drift_detector.get_active_alerts()
                
                for alert in active_alerts:
                    # Check if drift alert is relevant to the error
                    if self._is_drift_relevant_to_error(alert, error):
                        cause = PredictiveRootCause(
                            cause_type="data_drift",
                            description=f"Data drift detected in {alert.metric_name}",
                            confidence=alert.confidence,
                            contributing_factors=[alert.drift_type.value],
                            detection_method="drift_detection",
                            time_to_impact=alert.timestamp - time.time() if alert.timestamp > time.time() else 0,
                            preventive_measures=[
                                "Retrain model with recent data",
                                "Adjust drift detection thresholds",
                                "Implement data validation pipeline"
                            ]
                        )
                        drift_causes.append(cause)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing drift root causes: {e}")
        
        return drift_causes
    
    def _analyze_pattern_root_causes(
        self, 
        error: VoiceAIException, 
        system_state: Dict[str, Any]
    ) -> List[PredictiveRootCause]:
        """Analyze pattern-based root causes"""
        pattern_causes = []
        
        try:
            # Analyze error timing patterns
            if 'metrics' in system_state:
                metrics = system_state['metrics']
                
                # High latency pattern
                if hasattr(metrics, 'latency') and metrics.latency.get_p95() > 1000:
                    cause = PredictiveRootCause(
                        cause_type="high_latency",
                        description="High system latency detected",
                        confidence=0.8,
                        contributing_factors=["resource_exhaustion", "network_issues"],
                        detection_method="pattern_analysis",
                        time_to_impact=0,
                        preventive_measures=[
                            "Scale compute resources",
                            "Optimize query performance",
                            "Enable caching"
                        ]
                    )
                    pattern_causes.append(cause)
                
                # High error rate pattern
                if hasattr(metrics, 'throughput') and metrics.throughput.error_rate > 0.1:
                    cause = PredictiveRootCause(
                        cause_type="high_error_rate",
                        description="Elevated error rate detected",
                        confidence=0.9,
                        contributing_factors=["service_degradation", "configuration_issues"],
                        detection_method="pattern_analysis",
                        time_to_impact=0,
                        preventive_measures=[
                            "Review service health",
                            "Check configuration changes",
                            "Implement circuit breakers"
                        ]
                    )
                    pattern_causes.append(cause)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing pattern root causes: {e}")
        
        return pattern_causes
    
    async def _analyze_anomaly_root_causes(
        self, 
        error: VoiceAIException, 
        system_state: Dict[str, Any]
    ) -> List[PredictiveRootCause]:
        """Analyze anomaly-based root causes using predictive analytics"""
        anomaly_causes = []
        
        try:
            if self.predictive_analytics:
                # Get anomaly detection results
                anomalies = await self.predictive_analytics.detect_anomalies()
                
                for anomaly in anomalies:
                    if anomaly.get('severity', 'low') in ['high', 'critical']:
                        cause = PredictiveRootCause(
                            cause_type="anomaly_detection",
                            description=f"Anomaly detected: {anomaly.get('description', 'Unknown anomaly')}",
                            confidence=anomaly.get('confidence', 0.5),
                            contributing_factors=anomaly.get('factors', []),
                            detection_method="anomaly_detection",
                            time_to_impact=anomaly.get('time_to_impact', 0),
                            preventive_measures=anomaly.get('recommendations', [])
                        )
                        anomaly_causes.append(cause)
        
        except Exception as e:
            self.logger.warning(f"Error analyzing anomaly root causes: {e}")
        
        return anomaly_causes
    
    def _get_standard_recovery_recommendations(self, error: VoiceAIException) -> List[RecoveryRecommendation]:
        """Get standard recovery recommendations by error type"""
        recommendations = []
        
        error_type = error.__class__.__name__
        
        recovery_map = {
            'STTException': [
                RecoveryRecommendation(
                    action_type="service_restart",
                    description="Restart STT service",
                    priority="medium",
                    success_probability=0.7,
                    estimated_time=30,
                    automation_possible=True,
                    dependencies=["service_manager"],
                    risk_level="low",
                    rollback_plan="Revert to backup STT service"
                )
            ],
            'LLMException': [
                RecoveryRecommendation(
                    action_type="model_fallback",
                    description="Switch to backup LLM model",
                    priority="high",
                    success_probability=0.8,
                    estimated_time=5,
                    automation_possible=True,
                    dependencies=["model_router"],
                    risk_level="low"
                )
            ],
            'TTSException': [
                RecoveryRecommendation(
                    action_type="voice_fallback",
                    description="Switch to backup TTS voice",
                    priority="medium",
                    success_probability=0.9,
                    estimated_time=2,
                    automation_possible=True,
                    dependencies=["tts_manager"],
                    risk_level="low"
                )
            ],
            'AudioException': [
                RecoveryRecommendation(
                    action_type="audio_device_reset",
                    description="Reset audio device",
                    priority="high",
                    success_probability=0.6,
                    estimated_time=10,
                    automation_possible=True,
                    dependencies=["audio_driver"],
                    risk_level="medium"
                )
            ]
        }
        
        return recovery_map.get(error_type, [])
    
    def _calculate_pattern_similarity(self, error: VoiceAIException, pattern: ErrorPattern) -> float:
        """Calculate similarity between error and historical pattern"""
        try:
            similarity_factors = []
            
            # Error type similarity
            if error.__class__.__name__ == pattern.error_type:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            # Category similarity
            error_category = error.category.value if hasattr(error, 'category') else 'unknown'
            pattern_category = pattern.common_context.get('category', 'unknown')
            if error_category == pattern_category:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            # Context similarity (if available)
            if hasattr(error, 'context') and error.context:
                context_similarity = 0.0
                context_matches = 0
                total_context_keys = 0
                
                for key in ['service_name', 'operation', 'model_used']:
                    if key in error.context and key in pattern.common_context:
                        total_context_keys += 1
                        if error.context[key] == pattern.common_context[key]:
                            context_matches += 1
                
                if total_context_keys > 0:
                    context_similarity = context_matches / total_context_keys
                
                similarity_factors.append(context_similarity)
            
            # Calculate weighted average
            if similarity_factors:
                return sum(similarity_factors) / len(similarity_factors)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_incident_similarity(self, error: VoiceAIException, incident: Dict[str, Any]) -> float:
        """Calculate similarity between current error and historical incident"""
        try:
            similarity_factors = []
            
            # Error message similarity (simple approach)
            error_message = str(error).lower()
            incident_message = incident.get('message', '').lower()
            
            if error_message and incident_message:
                # Simple word overlap similarity
                error_words = set(error_message.split())
                incident_words = set(incident_message.split())
                
                if error_words and incident_words:
                    overlap = len(error_words.intersection(incident_words))
                    total = len(error_words.union(incident_words))
                    message_similarity = overlap / total if total > 0 else 0.0
                    similarity_factors.append(message_similarity)
            
            # Category similarity
            error_category = error.category.value if hasattr(error, 'category') else 'unknown'
            incident_category = incident.get('category', 'unknown')
            
            if error_category == incident_category:
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)
            
            # Calculate weighted average
            if similarity_factors:
                return sum(similarity_factors) / len(similarity_factors)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _is_automatable(self, action_description: str) -> bool:
        """Determine if an action can be automated"""
        automatable_keywords = [
            'restart', 'scale', 'switch', 'enable', 'disable', 
            'adjust', 'reset', 'clear', 'refresh', 'reload'
        ]
        
        description_lower = action_description.lower()
        return any(keyword in description_lower for keyword in automatable_keywords)
    
    def _deduplicate_recommendations(self, recommendations: List[RecoveryRecommendation]) -> List[RecoveryRecommendation]:
        """Remove duplicate recommendations"""
        seen_descriptions = set()
        unique_recommendations = []
        
        for recommendation in recommendations:
            desc_key = recommendation.description.lower().strip()
            if desc_key not in seen_descriptions:
                seen_descriptions.add(desc_key)
                unique_recommendations.append(recommendation)
        
        return unique_recommendations
    
    def _analyze_error_correlations(self, error: VoiceAIException) -> Tuple[List[str], Optional[str]]:
        """Analyze error correlations and clustering"""
        correlated_errors = []
        cluster_id = None
        
        try:
            # Simple correlation analysis based on time windows
            error_time = time.time()
            correlation_window = self.config['correlation_window_hours'] * 3600
            
            with self._lock:
                recent_errors = [
                    err for err in self.error_history 
                    if abs(err.get('timestamp', 0) - error_time) < correlation_window
                ]
                
                # Find errors with similar characteristics
                for err in recent_errors:
                    if self._are_errors_correlated(error, err):
                        correlated_errors.append(err.get('id', 'unknown'))
                
                # Simple clustering based on error type and time
                error_signature = f"{error.__class__.__name__}_{error.category.value if hasattr(error, 'category') else 'unknown'}"
                cluster_id = f"cluster_{hashlib.md5(error_signature.encode()).hexdigest()[:8]}"
                
        except Exception as e:
            self.logger.warning(f"Error analyzing correlations: {e}")
        
        return correlated_errors, cluster_id
    
    def _are_errors_correlated(self, error1: VoiceAIException, error2: Dict[str, Any]) -> bool:
        """Determine if two errors are correlated"""
        try:
            # Same error type
            if error1.__class__.__name__ == error2.get('type'):
                return True
            
            # Same category
            error1_category = error1.category.value if hasattr(error1, 'category') else 'unknown'
            if error1_category == error2.get('category'):
                return True
            
            # Same service/component
            if (hasattr(error1, 'context') and error1.context and 
                error1.context.get('service_name') == error2.get('service_name')):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _calculate_severity_escalation_risk(
        self, 
        error: VoiceAIException, 
        root_causes: List[PredictiveRootCause]
    ) -> float:
        """Calculate risk of severity escalation"""
        try:
            # Base risk from error category
            base_risk = {
                ErrorCategory.TRANSIENT: 0.2,
                ErrorCategory.PERMANENT: 0.8,
                ErrorCategory.RATE_LIMIT: 0.4,
                ErrorCategory.TIMEOUT: 0.3,
                ErrorCategory.AUTHENTICATION: 0.6,
                ErrorCategory.QUOTA: 0.5,
                ErrorCategory.NETWORK: 0.4,
                ErrorCategory.AUDIO: 0.3,
                ErrorCategory.MODEL: 0.7
            }.get(getattr(error, 'category', ErrorCategory.TRANSIENT), 0.5)
            
            # Adjust based on root causes
            cause_risk_adjustment = 0.0
            for cause in root_causes:
                if cause.confidence > 0.8:
                    cause_risk_adjustment += 0.2
                elif cause.confidence > 0.6:
                    cause_risk_adjustment += 0.1
            
            # Adjust based on correlation
            correlation_count = len(self._analyze_error_correlations(error)[0])
            correlation_adjustment = min(0.3, correlation_count * 0.1)
            
            final_risk = min(1.0, base_risk + cause_risk_adjustment + correlation_adjustment)
            return final_risk
            
        except Exception:
            return 0.5
    
    def _calculate_user_impact_score(self, error: VoiceAIException, system_state: Dict[str, Any]) -> float:
        """Calculate user impact score"""
        try:
            # Base impact from error type
            base_impact = {
                'STTException': 0.8,  # High impact - user can't communicate
                'LLMException': 0.6,  # Medium-high impact - reduced functionality
                'TTSException': 0.7,  # High impact - user can't hear responses
                'AudioException': 0.9,  # Very high impact - complete audio failure
                'VoiceAIException': 0.5  # Default medium impact
            }.get(error.__class__.__name__, 0.5)
            
            # Adjust based on system load
            if 'metrics' in system_state:
                metrics = system_state['metrics']
                if hasattr(metrics, 'throughput'):
                    current_load = getattr(metrics.throughput, 'requests_per_second', 0)
                    if current_load > 10:  # High load
                        base_impact *= 1.5
                    elif current_load > 5:  # Medium load
                        base_impact *= 1.2
            
            return min(1.0, base_impact)
            
        except Exception:
            return 0.5
    
    def _assess_business_impact(self, error: VoiceAIException, severity_risk: float, user_impact: float) -> str:
        """Assess business impact level"""
        try:
            combined_score = (severity_risk + user_impact) / 2
            
            if combined_score >= 0.8:
                return "critical"
            elif combined_score >= 0.6:
                return "high"
            elif combined_score >= 0.4:
                return "medium"
            else:
                return "low"
                
        except Exception:
            return "unknown"
    
    async def _get_leading_indicators(self, error: VoiceAIException) -> List[Dict[str, Any]]:
        """Get leading indicators that may have predicted this error"""
        indicators = []
        
        try:
            # Check drift indicators
            if self.drift_detector and self.config['drift_correlation_enabled']:
                recent_alerts = await self.drift_detector.get_recent_alerts(hours=1)
                for alert in recent_alerts:
                    indicators.append({
                        'type': 'drift_alert',
                        'description': f"Drift in {alert.metric_name}",
                        'time_before_error': time.time() - alert.timestamp,
                        'confidence': alert.confidence
                    })
            
            # Check performance indicators
            if self.metrics_collector:
                performance_alerts = self.metrics_collector.get_recent_alerts(hours=1)
                for alert in performance_alerts:
                    indicators.append({
                        'type': 'performance_alert',
                        'description': f"Performance issue: {alert.get('description', 'Unknown')}",
                        'time_before_error': time.time() - alert.get('timestamp', time.time()),
                        'severity': alert.get('severity', 'medium')
                    })
        
        except Exception as e:
            self.logger.warning(f"Error getting leading indicators: {e}")
        
        return indicators
    
    async def _build_degradation_timeline(self, error: VoiceAIException) -> List[Dict[str, Any]]:
        """Build timeline of system degradation leading to error"""
        timeline = []
        
        try:
            # Add error occurrence
            timeline.append({
                'timestamp': time.time(),
                'event_type': 'error_occurrence',
                'description': f"{error.__class__.__name__}: {str(error)}",
                'severity': 'high'
            })
            
            # Add recent system events if available
            if self.metrics_collector:
                recent_events = self.metrics_collector.get_recent_events(hours=0.5)
                for event in recent_events:
                    timeline.append({
                        'timestamp': event.get('timestamp', time.time()),
                        'event_type': 'system_event',
                        'description': event.get('description', 'System event'),
                        'severity': event.get('severity', 'medium')
                    })
            
            # Sort by timestamp
            timeline.sort(key=lambda x: x['timestamp'])
        
        except Exception as e:
            self.logger.warning(f"Error building degradation timeline: {e}")
        
        return timeline
    
    def _calculate_cascade_probability(self, error: VoiceAIException, correlated_errors: List[str]) -> float:
        """Calculate probability of error cascade"""
        try:
            # Base probability from error category
            base_probability = {
                ErrorCategory.TRANSIENT: 0.1,
                ErrorCategory.PERMANENT: 0.7,
                ErrorCategory.RATE_LIMIT: 0.8,
                ErrorCategory.TIMEOUT: 0.4,
                ErrorCategory.AUTHENTICATION: 0.9,
                ErrorCategory.QUOTA: 0.6,
                ErrorCategory.NETWORK: 0.5,
                ErrorCategory.AUDIO: 0.2,
                ErrorCategory.MODEL: 0.6
            }.get(getattr(error, 'category', ErrorCategory.TRANSIENT), 0.3)
            
            # Adjust based on correlated errors
            correlation_factor = min(0.4, len(correlated_errors) * 0.1)
            
            return min(1.0, base_probability + correlation_factor)
            
        except Exception:
            return 0.3
    
    def _calculate_enrichment_confidence(
        self,
        root_causes: List[PredictiveRootCause],
        similar_incidents: List[SimilarIncident],
        recommendations: List[RecoveryRecommendation]
    ) -> float:
        """Calculate overall confidence in enrichment results"""
        try:
            confidence_factors = []
            
            # Root cause confidence
            if root_causes:
                avg_cause_confidence = sum(c.confidence for c in root_causes) / len(root_causes)
                confidence_factors.append(avg_cause_confidence)
            
            # Similar incident confidence
            if similar_incidents:
                avg_incident_similarity = sum(i.context_similarity for i in similar_incidents) / len(similar_incidents)
                confidence_factors.append(avg_incident_similarity)
            
            # Recommendation confidence
            if recommendations:
                avg_rec_confidence = sum(r.success_probability for r in recommendations) / len(recommendations)
                confidence_factors.append(avg_rec_confidence)
            
            # Data availability factor
            data_availability = len(confidence_factors) / 3.0  # Expected 3 factors
            confidence_factors.append(data_availability)
            
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_prediction_reliability(self) -> float:
        """Calculate reliability of prediction systems"""
        try:
            reliability_factors = []
            
            # Drift detector reliability
            if self.drift_detector:
                reliability_factors.append(0.8)  # Assume good reliability
            
            # Predictive analytics reliability
            if self.predictive_analytics:
                reliability_factors.append(0.7)  # Assume good reliability
            
            # Metrics collector reliability
            if self.metrics_collector:
                reliability_factors.append(0.9)  # High reliability for metrics
            
            if reliability_factors:
                return sum(reliability_factors) / len(reliability_factors)
            else:
                return 0.5
                
        except Exception:
            return 0.5
    
    def _calculate_data_quality_score(self, system_state: Dict[str, Any]) -> float:
        """Calculate quality score of captured data"""
        try:
            quality_factors = []
            
            # System state completeness
            expected_keys = ['timestamp', 'metrics', 'analytics_state']
            present_keys = [key for key in expected_keys if key in system_state]
            completeness = len(present_keys) / len(expected_keys)
            quality_factors.append(completeness)
            
            # Data freshness
            state_age = time.time() - system_state.get('timestamp', time.time())
            freshness = max(0.0, 1.0 - (state_age / 300))  # 5 minute freshness window
            quality_factors.append(freshness)
            
            # Error presence (lower is better)
            error_penalty = 0.3 if 'capture_error' in system_state else 0.0
            quality_factors.append(1.0 - error_penalty)
            
            return sum(quality_factors) / len(quality_factors)
            
        except Exception:
            return 0.5
    
    def _get_active_components(self) -> List[str]:
        """Get list of currently active system components"""
        components = []
        
        if self.predictive_analytics:
            components.append("predictive_analytics")
        if self.metrics_collector:
            components.append("metrics_collector")
        if self.drift_detector:
            components.append("drift_detector")
        
        return components
    
    def _is_drift_relevant_to_error(self, alert, error: VoiceAIException) -> bool:
        """Check if drift alert is relevant to the error"""
        try:
            # Check if error category matches drift metric
            error_category = error.category.value if hasattr(error, 'category') else 'unknown'
            
            # Simple relevance mapping
            relevance_map = {
                'audio': ['AudioException'],
                'stt': ['STTException'],
                'llm': ['LLMException'],
                'tts': ['TTSException']
            }
            
            metric_type = alert.metric_name.lower()
            for key, error_types in relevance_map.items():
                if key in metric_type and error.__class__.__name__ in error_types:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _create_minimal_enrichment(self, error: VoiceAIException, duration_ms: float) -> EnrichedErrorContext:
        """Create minimal enrichment when full enrichment fails"""
        return EnrichedErrorContext(
            original_error=error,
            enrichment_timestamp=time.time(),
            enrichment_duration_ms=duration_ms,
            predicted_root_causes=[],
            similar_incidents=[],
            recovery_recommendations=[],
            error_patterns=[],
            system_state_at_error={'minimal': True, 'timestamp': time.time()},
            leading_indicators=[],
            degradation_timeline=[],
            correlated_errors=[],
            error_cluster_id=None,
            cascade_probability=0.0,
            severity_escalation_risk=0.5,
            user_impact_score=0.5,
            business_impact_assessment="unknown",
            enrichment_confidence=0.0,
            prediction_reliability=0.0,
            data_quality_score=0.0
        )
    
    def _store_error_for_analysis(self, error: VoiceAIException, enrichment: EnrichedErrorContext):
        """Store error in history for future pattern analysis"""
        try:
            error_record = {
                'id': f"error_{int(time.time())}_{id(error)}",
                'timestamp': time.time(),
                'type': error.__class__.__name__,
                'category': error.category.value if hasattr(error, 'category') else 'unknown',
                'message': str(error),
                'enrichment_confidence': enrichment.enrichment_confidence,
                'resolution_method': None,  # To be updated when resolved
                'resolution_success': None,  # To be updated when resolved
                'lessons_learned': []  # To be updated when resolved
            }
            
            if hasattr(error, 'context') and error.context:
                error_record['context'] = error.context
            
            with self._lock:
                self.error_history.append(error_record)
                
        except Exception as e:
            self.logger.warning(f"Failed to store error for analysis: {e}")
    
    def _update_enrichment_stats(self, duration_ms: float, success: bool):
        """Update enrichment performance statistics"""
        with self._lock:
            self.enrichment_stats['total_enrichments'] += 1
            
            if success:
                self.enrichment_stats['successful_enrichments'] += 1
            
            # Update average duration
            total = self.enrichment_stats['total_enrichments']
            current_avg = self.enrichment_stats['average_enrichment_time_ms']
            new_avg = ((current_avg * (total - 1)) + duration_ms) / total
            self.enrichment_stats['average_enrichment_time_ms'] = new_avg
    
    def get_enrichment_statistics(self) -> Dict[str, Any]:
        """Get enrichment performance statistics"""
        with self._lock:
            stats = self.enrichment_stats.copy()
            
            if stats['total_enrichments'] > 0:
                stats['success_rate'] = stats['successful_enrichments'] / stats['total_enrichments']
            else:
                stats['success_rate'] = 0.0
            
            return stats
    
    async def start_async_processing(self):
        """Start asynchronous error enrichment processing"""
        if self.config['async_enrichment_enabled']:
            # Start background task for processing enrichment queue
            task = asyncio.create_task(self._process_enrichment_queue())
            self.enrichment_tasks.append(task)
            self.logger.info("🔍 Async error enrichment processing started")
    
    async def stop_async_processing(self):
        """Stop asynchronous error enrichment processing"""
        # Cancel all enrichment tasks
        for task in self.enrichment_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.enrichment_tasks:
            await asyncio.gather(*self.enrichment_tasks, return_exceptions=True)
        
        self.enrichment_tasks.clear()
        self.logger.info("🔍 Async error enrichment processing stopped")
    
    async def _process_enrichment_queue(self):
        """Process the async enrichment queue"""
        try:
            while True:
                # Get error from queue with timeout
                try:
                    error, future = await asyncio.wait_for(
                        self.enrichment_queue.get(), 
                        timeout=1.0
                    )
                    
                    # Perform enrichment
                    try:
                        enrichment = await self._perform_enrichment(error)
                        future.set_result(enrichment)
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        self.enrichment_queue.task_done()
                        
                except asyncio.TimeoutError:
                    # No items in queue, continue
                    continue
                    
        except asyncio.CancelledError:
            self.logger.info("🔍 Enrichment queue processing cancelled")
        except Exception as e:
            self.logger.error(f"Error in enrichment queue processing: {e}")

# Factory function for easy instantiation
def create_predictive_error_enricher(
    predictive_analytics: Optional[PredictiveAnalyticsFramework] = None,
    metrics_collector: Optional[MetricsCollector] = None,
    drift_detector: Optional[DriftDetector] = None,
    logger: Optional[VoiceAILogger] = None,
    config: Optional[Dict[str, Any]] = None
) -> PredictiveErrorEnricher:
    """Create and configure a PredictiveErrorEnricher instance"""
    return PredictiveErrorEnricher(
        predictive_analytics=predictive_analytics,
        metrics_collector=metrics_collector,
        drift_detector=drift_detector,
        logger=logger,
        config=config
    )

# Global instance management
_predictive_error_enricher: Optional[PredictiveErrorEnricher] = None

def get_predictive_error_enricher() -> Optional[PredictiveErrorEnricher]:
    """Get the global predictive error enricher instance"""
    return _predictive_error_enricher

def set_predictive_error_enricher(enricher: PredictiveErrorEnricher):
    """Set the global predictive error enricher instance"""
    global _predictive_error_enricher
    _predictive_error_enricher = enricher 