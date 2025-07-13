"""
Automated Report Generation for Sovereign 4.0 Voice Assistant Performance Testing

This module provides comprehensive report generation capabilities including:
- Automated performance test reports with pass/fail criteria
- Actionable insights and recommendations
- Performance trend analysis
- Executive summaries and detailed technical reports
- Export to multiple formats (JSON, HTML, PDF)

Implements modern 2024-2025 performance reporting best practices.
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import base64

from .performance_testing import TestResult, PerformanceTestReport, PerformanceTestConfig

logger = logging.getLogger(__name__)

@dataclass
class PerformanceInsight:
    """Individual performance insight or recommendation"""
    insight_id: str
    category: str  # 'critical', 'warning', 'optimization', 'information'
    title: str
    description: str
    impact: str  # 'high', 'medium', 'low'
    effort: str  # 'high', 'medium', 'low'
    recommendations: List[str]
    affected_components: List[str]
    metrics_evidence: Dict[str, float]

@dataclass
class ExecutiveSummary:
    """Executive summary of performance test results"""
    overall_status: str
    test_duration_hours: float
    total_tests_run: int
    tests_passed: int
    tests_failed: int
    tests_with_warnings: int
    key_findings: List[str]
    critical_issues: List[str]
    recommendations_summary: List[str]
    performance_score: float  # 0-100 composite score

class ReportGenerator:
    """
    Comprehensive report generator for performance testing results
    
    Creates detailed reports with actionable insights, trend analysis,
    and specific recommendations for performance optimization.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        self.report_storage_path = Path('.performance_reports')
        self.report_storage_path.mkdir(exist_ok=True)
        
        # Performance scoring weights
        self.scoring_weights = {
            'latency': 0.3,
            'accuracy': 0.25,
            'reliability': 0.2,
            'resource_efficiency': 0.15,
            'stability': 0.1
        }
        
        logger.info("Report Generator initialized")
    
    async def generate_report(self, 
                            session_id: str,
                            test_results: List[TestResult],
                            execution_time: float) -> PerformanceTestReport:
        """Generate comprehensive performance test report"""
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(test_results)
        
        # Generate insights and recommendations
        insights = await self._generate_insights(test_results, summary_metrics)
        recommendations = self._extract_recommendations(insights)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            test_results, execution_time, insights
        )
        
        # Determine overall status
        overall_status = self._determine_overall_status(test_results, insights)
        
        # Create comprehensive report
        report = PerformanceTestReport(
            test_session_id=session_id,
            timestamp=datetime.now(),
            overall_status=overall_status,
            test_results=test_results,
            summary_metrics=summary_metrics,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        # Save report to storage
        await self._save_report(report, executive_summary, insights)
        
        # Generate additional report formats
        await self._generate_report_formats(report, executive_summary, insights)
        
        logger.info(f"Performance report generated: {session_id} ({overall_status})")
        return report
    
    async def generate_continuous_monitoring_report(self,
                                                  session_id: str,
                                                  test_results: List[TestResult],
                                                  execution_time: float,
                                                  duration_hours: int) -> PerformanceTestReport:
        """Generate specialized report for continuous monitoring"""
        
        # Enhanced metrics for continuous monitoring
        summary_metrics = self._calculate_summary_metrics(test_results)
        
        # Add continuous monitoring specific metrics
        continuous_metrics = self._calculate_continuous_monitoring_metrics(
            test_results, duration_hours
        )
        summary_metrics.update(continuous_metrics)
        
        # Generate specialized insights for long-running tests
        insights = await self._generate_continuous_monitoring_insights(
            test_results, summary_metrics, duration_hours
        )
        
        recommendations = self._extract_recommendations(insights)
        
        # Create executive summary
        executive_summary = self._create_executive_summary(
            test_results, execution_time, insights, continuous_monitoring=True
        )
        
        overall_status = self._determine_overall_status(test_results, insights)
        
        report = PerformanceTestReport(
            test_session_id=session_id,
            timestamp=datetime.now(),
            overall_status=overall_status,
            test_results=test_results,
            summary_metrics=summary_metrics,
            recommendations=recommendations,
            execution_time=execution_time
        )
        
        # Save with continuous monitoring designation
        await self._save_continuous_monitoring_report(
            report, executive_summary, insights, duration_hours
        )
        
        logger.info(f"Continuous monitoring report generated: {session_id} ({duration_hours}h)")
        return report
    
    def _calculate_summary_metrics(self, test_results: List[TestResult]) -> Dict[str, float]:
        """Calculate comprehensive summary metrics"""
        if not test_results:
            return {}
        
        # Basic test statistics
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == 'passed'])
        failed_tests = len([r for r in test_results if r.status == 'failed'])
        warning_tests = len([r for r in test_results if r.status == 'warning'])
        skipped_tests = len([r for r in test_results if r.status == 'skipped'])
        
        summary = {
            'total_tests': total_tests,
            'tests_passed': passed_tests,
            'tests_failed': failed_tests,
            'tests_warning': warning_tests,
            'tests_skipped': skipped_tests,
            'pass_rate_percent': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'fail_rate_percent': (failed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_execution_time': sum(r.execution_time for r in test_results)
        }
        
        # Aggregate performance metrics
        latency_metrics = []
        accuracy_metrics = []
        memory_metrics = []
        throughput_metrics = []
        
        for result in test_results:
            if result.metrics:
                # Collect latency metrics
                for key, value in result.metrics.items():
                    if 'latency' in key.lower() and 'ms' in key:
                        latency_metrics.append(value)
                    elif 'accuracy' in key.lower() or 'score' in key:
                        accuracy_metrics.append(value)
                    elif 'memory' in key.lower() and ('mb' in key or 'percent' in key):
                        memory_metrics.append(value)
                    elif 'throughput' in key.lower() or 'rps' in key:
                        throughput_metrics.append(value)
        
        # Calculate aggregated metrics
        if latency_metrics:
            summary.update({
                'avg_latency_ms': sum(latency_metrics) / len(latency_metrics),
                'max_latency_ms': max(latency_metrics),
                'min_latency_ms': min(latency_metrics)
            })
        
        if accuracy_metrics:
            summary.update({
                'avg_accuracy': sum(accuracy_metrics) / len(accuracy_metrics),
                'min_accuracy': min(accuracy_metrics),
                'max_accuracy': max(accuracy_metrics)
            })
        
        if memory_metrics:
            summary.update({
                'avg_memory_usage': sum(memory_metrics) / len(memory_metrics),
                'peak_memory_usage': max(memory_metrics)
            })
        
        if throughput_metrics:
            summary.update({
                'avg_throughput': sum(throughput_metrics) / len(throughput_metrics),
                'peak_throughput': max(throughput_metrics)
            })
        
        # Calculate composite performance score
        summary['performance_score'] = self._calculate_performance_score(summary, test_results)
        
        return summary
    
    def _calculate_continuous_monitoring_metrics(self, 
                                               test_results: List[TestResult],
                                               duration_hours: int) -> Dict[str, float]:
        """Calculate metrics specific to continuous monitoring"""
        continuous_metrics = {
            'monitoring_duration_hours': duration_hours,
            'tests_per_hour': len(test_results) / duration_hours if duration_hours > 0 else 0
        }
        
        # Find memory-related results
        memory_results = [r for r in test_results if 'memory' in r.test_name.lower()]
        if memory_results:
            memory_leaks_detected = sum(
                1 for r in memory_results 
                if r.metrics and r.metrics.get('leaks_detected', 0) > 0
            )
            continuous_metrics['memory_leaks_detected'] = memory_leaks_detected
        
        # Find regression results
        regression_results = [r for r in test_results if 'regression' in r.test_name.lower()]
        if regression_results:
            regressions_detected = sum(
                1 for r in regression_results
                if r.status in ['failed', 'warning']
            )
            continuous_metrics['regressions_detected'] = regressions_detected
        
        # Stability analysis
        failed_tests_over_time = [r for r in test_results if r.status == 'failed']
        continuous_metrics['stability_issues'] = len(failed_tests_over_time)
        
        return continuous_metrics
    
    async def _generate_insights(self, 
                               test_results: List[TestResult],
                               summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Generate comprehensive performance insights"""
        insights = []
        
        # Analyze latency performance
        insights.extend(await self._analyze_latency_insights(test_results, summary_metrics))
        
        # Analyze accuracy performance
        insights.extend(await self._analyze_accuracy_insights(test_results, summary_metrics))
        
        # Analyze memory usage
        insights.extend(await self._analyze_memory_insights(test_results, summary_metrics))
        
        # Analyze system stability
        insights.extend(await self._analyze_stability_insights(test_results, summary_metrics))
        
        # Analyze resource efficiency
        insights.extend(await self._analyze_resource_insights(test_results, summary_metrics))
        
        # Cross-cutting performance analysis
        insights.extend(await self._analyze_overall_performance(test_results, summary_metrics))
        
        return insights
    
    async def _generate_continuous_monitoring_insights(self,
                                                     test_results: List[TestResult],
                                                     summary_metrics: Dict[str, float],
                                                     duration_hours: int) -> List[PerformanceInsight]:
        """Generate insights specific to continuous monitoring"""
        insights = await self._generate_insights(test_results, summary_metrics)
        
        # Add continuous monitoring specific insights
        continuous_insights = []
        
        # Long-term stability analysis
        if duration_hours >= 8:
            stability_insight = await self._analyze_long_term_stability(
                test_results, summary_metrics, duration_hours
            )
            if stability_insight:
                continuous_insights.append(stability_insight)
        
        # Memory leak analysis
        memory_leak_insight = await self._analyze_memory_leak_patterns(
            test_results, duration_hours
        )
        if memory_leak_insight:
            continuous_insights.append(memory_leak_insight)
        
        # Performance drift analysis
        drift_insight = await self._analyze_performance_drift(
            test_results, duration_hours
        )
        if drift_insight:
            continuous_insights.append(drift_insight)
        
        insights.extend(continuous_insights)
        return insights
    
    async def _analyze_latency_insights(self, 
                                      test_results: List[TestResult],
                                      summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze latency performance and generate insights"""
        insights = []
        
        avg_latency = summary_metrics.get('avg_latency_ms', 0)
        max_latency = summary_metrics.get('max_latency_ms', 0)
        target_latency = self.config.latency_targets.get('cloud_p95', 800)
        
        # High latency insight
        if avg_latency > target_latency:
            insight = PerformanceInsight(
                insight_id=f"latency_high_{int(time.time())}",
                category='critical',
                title='High Average Latency Detected',
                description=f'Average latency ({avg_latency:.1f}ms) exceeds target ({target_latency}ms)',
                impact='high',
                effort='medium',
                recommendations=[
                    'Optimize STT processing pipeline',
                    'Review LLM model selection and routing',
                    'Implement response caching for common queries',
                    'Consider edge deployment for reduced network latency'
                ],
                affected_components=['stt_processing', 'llm_inference', 'tts_processing'],
                metrics_evidence={'avg_latency_ms': avg_latency, 'target_latency_ms': target_latency}
            )
            insights.append(insight)
        
        # Latency variability insight
        if max_latency > avg_latency * 2:
            insight = PerformanceInsight(
                insight_id=f"latency_variability_{int(time.time())}",
                category='warning',
                title='High Latency Variability',
                description=f'Maximum latency ({max_latency:.1f}ms) is significantly higher than average',
                impact='medium',
                effort='medium',
                recommendations=[
                    'Implement request timeout mechanisms',
                    'Add circuit breakers for external services',
                    'Monitor and optimize worst-case scenarios',
                    'Implement adaptive load balancing'
                ],
                affected_components=['overall_pipeline'],
                metrics_evidence={'max_latency_ms': max_latency, 'avg_latency_ms': avg_latency}
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_accuracy_insights(self,
                                       test_results: List[TestResult],
                                       summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze accuracy performance and generate insights"""
        insights = []
        
        avg_accuracy = summary_metrics.get('avg_accuracy', 0)
        min_accuracy = summary_metrics.get('min_accuracy', 0)
        
        # Low accuracy insight
        if avg_accuracy < 0.9:  # 90% threshold
            insight = PerformanceInsight(
                insight_id=f"accuracy_low_{int(time.time())}",
                category='critical',
                title='Below Target Accuracy Performance',
                description=f'Average accuracy ({avg_accuracy:.3f}) is below acceptable threshold',
                impact='high',
                effort='high',
                recommendations=[
                    'Review and retrain models with better data',
                    'Implement ensemble methods for improved accuracy',
                    'Fine-tune model parameters',
                    'Validate input data quality and preprocessing'
                ],
                affected_components=['stt_transcription', 'memory_recall', 'llm_processing'],
                metrics_evidence={'avg_accuracy': avg_accuracy, 'target_accuracy': 0.9}
            )
            insights.append(insight)
        
        # Accuracy consistency insight
        if min_accuracy < avg_accuracy * 0.8:  # More than 20% below average
            insight = PerformanceInsight(
                insight_id=f"accuracy_consistency_{int(time.time())}",
                category='warning',
                title='Inconsistent Accuracy Performance',
                description='Significant variation in accuracy across different test scenarios',
                impact='medium',
                effort='medium',
                recommendations=[
                    'Identify and address edge cases',
                    'Implement confidence scoring and fallback mechanisms',
                    'Add input validation and preprocessing improvements',
                    'Consider specialized models for different use cases'
                ],
                affected_components=['model_inference'],
                metrics_evidence={'min_accuracy': min_accuracy, 'avg_accuracy': avg_accuracy}
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_memory_insights(self,
                                     test_results: List[TestResult],
                                     summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze memory usage and generate insights"""
        insights = []
        
        # Find memory-related test results
        memory_results = [r for r in test_results if 'memory' in r.test_name.lower()]
        
        for result in memory_results:
            if result.metrics:
                memory_growth = result.metrics.get('memory_growth_percent', 0)
                leaks_detected = result.metrics.get('leaks_detected', 0)
                
                # Memory leak insight
                if leaks_detected > 0 or memory_growth > 10:
                    insight = PerformanceInsight(
                        insight_id=f"memory_leak_{int(time.time())}",
                        category='critical' if leaks_detected > 0 else 'warning',
                        title='Memory Leak Detected' if leaks_detected > 0 else 'High Memory Growth',
                        description=f'Memory growth: {memory_growth:.1f}%, Leaks detected: {leaks_detected}',
                        impact='high',
                        effort='high',
                        recommendations=[
                            'Implement proper resource cleanup in audio processing',
                            'Review object lifecycle management',
                            'Add memory profiling to development process',
                            'Implement garbage collection optimization'
                        ],
                        affected_components=['audio_processing', 'model_inference'],
                        metrics_evidence={'memory_growth_percent': memory_growth, 'leaks_detected': leaks_detected}
                    )
                    insights.append(insight)
        
        return insights
    
    async def _analyze_stability_insights(self,
                                        test_results: List[TestResult],
                                        summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze system stability and generate insights"""
        insights = []
        
        fail_rate = summary_metrics.get('fail_rate_percent', 0)
        
        # High failure rate insight
        if fail_rate > 10:  # More than 10% failure rate
            insight = PerformanceInsight(
                insight_id=f"stability_low_{int(time.time())}",
                category='critical',
                title='High Test Failure Rate',
                description=f'Test failure rate ({fail_rate:.1f}%) indicates stability issues',
                impact='high',
                effort='high',
                recommendations=[
                    'Implement comprehensive error handling',
                    'Add retry mechanisms for transient failures',
                    'Improve input validation and sanitization',
                    'Enhance monitoring and alerting systems'
                ],
                affected_components=['overall_system'],
                metrics_evidence={'fail_rate_percent': fail_rate}
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_resource_insights(self,
                                       test_results: List[TestResult],
                                       summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze resource efficiency and generate insights"""
        insights = []
        
        peak_memory = summary_metrics.get('peak_memory_usage', 0)
        
        # High resource usage insight
        if peak_memory > 1000:  # More than 1GB peak memory
            insight = PerformanceInsight(
                insight_id=f"resource_high_{int(time.time())}",
                category='warning',
                title='High Resource Usage',
                description=f'Peak memory usage ({peak_memory:.1f}MB) is high',
                impact='medium',
                effort='medium',
                recommendations=[
                    'Optimize model loading and caching strategies',
                    'Implement memory-efficient data processing',
                    'Consider model quantization for reduced memory footprint',
                    'Add resource monitoring and limits'
                ],
                affected_components=['model_inference', 'audio_processing'],
                metrics_evidence={'peak_memory_mb': peak_memory}
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_overall_performance(self,
                                         test_results: List[TestResult],
                                         summary_metrics: Dict[str, float]) -> List[PerformanceInsight]:
        """Analyze overall performance patterns"""
        insights = []
        
        performance_score = summary_metrics.get('performance_score', 0)
        
        # Overall performance insight
        if performance_score < 70:  # Below 70/100 score
            insight = PerformanceInsight(
                insight_id=f"performance_overall_{int(time.time())}",
                category='warning',
                title='Below Target Overall Performance',
                description=f'Composite performance score ({performance_score:.1f}/100) needs improvement',
                impact='high',
                effort='high',
                recommendations=[
                    'Prioritize optimization of critical performance paths',
                    'Implement performance budgets and monitoring',
                    'Regular performance regression testing',
                    'Consider architectural improvements for scalability'
                ],
                affected_components=['overall_system'],
                metrics_evidence={'performance_score': performance_score}
            )
            insights.append(insight)
        
        return insights
    
    async def _analyze_long_term_stability(self,
                                         test_results: List[TestResult],
                                         summary_metrics: Dict[str, float],
                                         duration_hours: int) -> Optional[PerformanceInsight]:
        """Analyze long-term stability patterns"""
        # Find tests that ran throughout the monitoring period
        time_distributed_failures = []
        
        for result in test_results:
            if result.status == 'failed' and result.timestamp:
                time_distributed_failures.append(result.timestamp)
        
        if len(time_distributed_failures) > duration_hours:  # More than 1 failure per hour
            return PerformanceInsight(
                insight_id=f"stability_longterm_{int(time.time())}",
                category='warning',
                title='Long-term Stability Concerns',
                description=f'Consistent failures detected over {duration_hours}h monitoring period',
                impact='medium',
                effort='medium',
                recommendations=[
                    'Implement graceful degradation mechanisms',
                    'Add automated recovery procedures',
                    'Review system architecture for resilience',
                    'Enhance monitoring and alerting for early detection'
                ],
                affected_components=['overall_system'],
                metrics_evidence={'duration_hours': duration_hours, 'failure_frequency': len(time_distributed_failures)}
            )
        
        return None
    
    async def _analyze_memory_leak_patterns(self,
                                          test_results: List[TestResult],
                                          duration_hours: int) -> Optional[PerformanceInsight]:
        """Analyze memory leak patterns over time"""
        memory_results = [r for r in test_results if 'memory' in r.test_name.lower()]
        
        if memory_results:
            total_leaks = sum(
                r.metrics.get('leaks_detected', 0) 
                for r in memory_results 
                if r.metrics
            )
            
            if total_leaks > 0:
                return PerformanceInsight(
                    insight_id=f"memory_pattern_{int(time.time())}",
                    category='critical',
                    title='Memory Leak Pattern Detected',
                    description=f'{total_leaks} memory leaks detected over {duration_hours}h',
                    impact='high',
                    effort='high',
                    recommendations=[
                        'Implement comprehensive memory leak detection in CI/CD',
                        'Add automated memory cleanup procedures',
                        'Review resource management in long-running processes',
                        'Implement memory usage limits and monitoring'
                    ],
                    affected_components=['memory_management'],
                    metrics_evidence={'total_leaks': total_leaks, 'duration_hours': duration_hours}
                )
        
        return None
    
    async def _analyze_performance_drift(self,
                                       test_results: List[TestResult],
                                       duration_hours: int) -> Optional[PerformanceInsight]:
        """Analyze performance drift over time"""
        regression_results = [r for r in test_results if 'regression' in r.test_name.lower()]
        
        if regression_results:
            regressions_detected = sum(
                1 for r in regression_results 
                if r.status in ['failed', 'warning']
            )
            
            if regressions_detected > 0:
                return PerformanceInsight(
                    insight_id=f"drift_pattern_{int(time.time())}",
                    category='warning',
                    title='Performance Drift Detected',
                    description=f'Performance regression detected over {duration_hours}h monitoring',
                    impact='medium',
                    effort='medium',
                    recommendations=[
                        'Implement continuous performance baselines',
                        'Add automated performance regression alerts',
                        'Review recent system changes and deployments',
                        'Establish performance SLA monitoring'
                    ],
                    affected_components=['performance_monitoring'],
                    metrics_evidence={'regressions_detected': regressions_detected}
                )
        
        return None
    
    def _calculate_performance_score(self, 
                                   summary_metrics: Dict[str, float],
                                   test_results: List[TestResult]) -> float:
        """Calculate composite performance score (0-100)"""
        scores = {}
        
        # Latency score (lower is better)
        avg_latency = summary_metrics.get('avg_latency_ms', 0)
        target_latency = self.config.latency_targets.get('cloud_p95', 800)
        if avg_latency > 0:
            scores['latency'] = max(0, 100 - (avg_latency / target_latency) * 100)
        else:
            scores['latency'] = 100
        
        # Accuracy score
        avg_accuracy = summary_metrics.get('avg_accuracy', 1.0)
        scores['accuracy'] = avg_accuracy * 100
        
        # Reliability score (based on pass rate)
        pass_rate = summary_metrics.get('pass_rate_percent', 100)
        scores['reliability'] = pass_rate
        
        # Resource efficiency score
        peak_memory = summary_metrics.get('peak_memory_usage', 0)
        if peak_memory > 0:
            # Score based on memory usage (lower is better)
            scores['resource_efficiency'] = max(0, 100 - (peak_memory / 2000) * 100)  # 2GB baseline
        else:
            scores['resource_efficiency'] = 100
        
        # Stability score (based on consistency)
        fail_rate = summary_metrics.get('fail_rate_percent', 0)
        scores['stability'] = max(0, 100 - fail_rate * 2)  # Double penalty for failures
        
        # Calculate weighted average
        weighted_score = sum(
            scores.get(component, 50) * weight 
            for component, weight in self.scoring_weights.items()
        )
        
        return min(100, max(0, weighted_score))
    
    def _extract_recommendations(self, insights: List[PerformanceInsight]) -> List[str]:
        """Extract top recommendations from insights"""
        all_recommendations = []
        
        # Prioritize critical and high-impact insights
        critical_insights = [i for i in insights if i.category == 'critical']
        high_impact_insights = [i for i in insights if i.impact == 'high']
        
        priority_insights = critical_insights + high_impact_insights
        
        for insight in priority_insights:
            all_recommendations.extend(insight.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)
        
        # Return top 10 recommendations
        return unique_recommendations[:10]
    
    def _create_executive_summary(self,
                                test_results: List[TestResult],
                                execution_time: float,
                                insights: List[PerformanceInsight],
                                continuous_monitoring: bool = False) -> ExecutiveSummary:
        """Create executive summary of test results"""
        
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.status == 'passed'])
        failed_tests = len([r for r in test_results if r.status == 'failed'])
        warning_tests = len([r for r in test_results if r.status == 'warning'])
        
        # Determine overall status
        if failed_tests == 0:
            overall_status = 'passed'
        elif failed_tests > total_tests * 0.2:  # More than 20% failures
            overall_status = 'failed'
        else:
            overall_status = 'warning'
        
        # Extract key findings
        key_findings = []
        critical_issues = []
        
        for insight in insights:
            if insight.category == 'critical':
                critical_issues.append(insight.title)
            elif insight.impact == 'high':
                key_findings.append(insight.title)
        
        # Add performance findings
        if execution_time > 3600:  # More than 1 hour
            key_findings.append(f"Extended testing completed successfully over {execution_time/3600:.1f} hours")
        
        # Calculate performance score
        summary_metrics = self._calculate_summary_metrics(test_results)
        performance_score = summary_metrics.get('performance_score', 0)
        
        # Generate recommendations summary
        recommendations_summary = self._extract_recommendations(insights)[:5]  # Top 5 for executive summary
        
        return ExecutiveSummary(
            overall_status=overall_status,
            test_duration_hours=execution_time / 3600,
            total_tests_run=total_tests,
            tests_passed=passed_tests,
            tests_failed=failed_tests,
            tests_with_warnings=warning_tests,
            key_findings=key_findings,
            critical_issues=critical_issues,
            recommendations_summary=recommendations_summary,
            performance_score=performance_score
        )
    
    def _determine_overall_status(self, 
                                test_results: List[TestResult],
                                insights: List[PerformanceInsight]) -> str:
        """Determine overall test status"""
        # Count test results
        failed_count = len([r for r in test_results if r.status == 'failed'])
        total_count = len(test_results)
        
        # Count critical insights
        critical_insights = len([i for i in insights if i.category == 'critical'])
        
        # Determine status based on failures and critical issues
        if failed_count == 0 and critical_insights == 0:
            return 'passed'
        elif failed_count > total_count * 0.2 or critical_insights > 2:  # 20% failure rate or multiple critical issues
            return 'failed'
        else:
            return 'partial'
    
    async def _save_report(self,
                         report: PerformanceTestReport,
                         executive_summary: ExecutiveSummary,
                         insights: List[PerformanceInsight]) -> None:
        """Save comprehensive report to storage"""
        try:
            # Create report data structure
            report_data = {
                'report': asdict(report),
                'executive_summary': asdict(executive_summary),
                'insights': [asdict(insight) for insight in insights],
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0'
            }
            
            # Save JSON report
            json_file = self.report_storage_path / f"{report.test_session_id}_report.json"
            with open(json_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Performance report saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    async def _save_continuous_monitoring_report(self,
                                               report: PerformanceTestReport,
                                               executive_summary: ExecutiveSummary,
                                               insights: List[PerformanceInsight],
                                               duration_hours: int) -> None:
        """Save continuous monitoring report with special designation"""
        try:
            # Enhanced report data for continuous monitoring
            report_data = {
                'report': asdict(report),
                'executive_summary': asdict(executive_summary),
                'insights': [asdict(insight) for insight in insights],
                'monitoring_type': 'continuous',
                'duration_hours': duration_hours,
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0'
            }
            
            # Save with continuous monitoring designation
            json_file = self.report_storage_path / f"{report.test_session_id}_continuous_{duration_hours}h_report.json"
            with open(json_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            logger.info(f"Continuous monitoring report saved: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to save continuous monitoring report: {e}")
    
    async def _generate_report_formats(self,
                                     report: PerformanceTestReport,
                                     executive_summary: ExecutiveSummary,
                                     insights: List[PerformanceInsight]) -> None:
        """Generate additional report formats (HTML, summary)"""
        try:
            # Generate HTML report
            await self._generate_html_report(report, executive_summary, insights)
            
            # Generate summary report
            await self._generate_summary_report(report, executive_summary)
            
        except Exception as e:
            logger.error(f"Failed to generate additional report formats: {e}")
    
    async def _generate_html_report(self,
                                  report: PerformanceTestReport,
                                  executive_summary: ExecutiveSummary,
                                  insights: List[PerformanceInsight]) -> None:
        """Generate HTML report for web viewing"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Test Report - {report.test_session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .status-passed {{ color: green; font-weight: bold; }}
                .status-failed {{ color: red; font-weight: bold; }}
                .status-warning {{ color: orange; font-weight: bold; }}
                .insight {{ margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }}
                .insight-critical {{ border-left-color: #dc3545; }}
                .insight-warning {{ border-left-color: #ffc107; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                .metric {{ background: #f8f9fa; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Performance Test Report</h1>
                <p><strong>Session ID:</strong> {report.test_session_id}</p>
                <p><strong>Generated:</strong> {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Status:</strong> <span class="status-{report.overall_status}">{report.overall_status.upper()}</span></p>
                <p><strong>Performance Score:</strong> {executive_summary.performance_score:.1f}/100</p>
            </div>
            
            <h2>Executive Summary</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>Total Tests:</strong> {executive_summary.total_tests_run}
                </div>
                <div class="metric">
                    <strong>Passed:</strong> {executive_summary.tests_passed}
                </div>
                <div class="metric">
                    <strong>Failed:</strong> {executive_summary.tests_failed}
                </div>
                <div class="metric">
                    <strong>Warnings:</strong> {executive_summary.tests_with_warnings}
                </div>
                <div class="metric">
                    <strong>Duration:</strong> {executive_summary.test_duration_hours:.2f}h
                </div>
            </div>
            
            <h2>Key Performance Metrics</h2>
            <div class="metrics">
        """
        
        for key, value in report.summary_metrics.items():
            if isinstance(value, (int, float)):
                html_content += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value:.2f}</div>'
        
        html_content += """
            </div>
            
            <h2>Critical Issues</h2>
        """
        
        if executive_summary.critical_issues:
            for issue in executive_summary.critical_issues:
                html_content += f'<div class="insight insight-critical">{issue}</div>'
        else:
            html_content += '<p>No critical issues detected.</p>'
        
        html_content += """
            <h2>Recommendations</h2>
            <ul>
        """
        
        for recommendation in executive_summary.recommendations_summary:
            html_content += f'<li>{recommendation}</li>'
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        # Save HTML report
        html_file = self.report_storage_path / f"{report.test_session_id}_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    async def _generate_summary_report(self,
                                     report: PerformanceTestReport,
                                     executive_summary: ExecutiveSummary) -> None:
        """Generate concise summary report"""
        summary_content = f"""
Performance Test Summary Report
==============================

Session ID: {report.test_session_id}
Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Status: {report.overall_status.upper()}
Performance Score: {executive_summary.performance_score:.1f}/100

Test Results:
- Total Tests: {executive_summary.total_tests_run}
- Passed: {executive_summary.tests_passed}
- Failed: {executive_summary.tests_failed}
- Warnings: {executive_summary.tests_with_warnings}
- Duration: {executive_summary.test_duration_hours:.2f} hours

Key Metrics:
"""
        
        key_metrics = ['avg_latency_ms', 'avg_accuracy', 'pass_rate_percent', 'peak_memory_usage']
        for metric in key_metrics:
            if metric in report.summary_metrics:
                value = report.summary_metrics[metric]
                summary_content += f"- {metric.replace('_', ' ').title()}: {value:.2f}\n"
        
        summary_content += "\nTop Recommendations:\n"
        for i, rec in enumerate(executive_summary.recommendations_summary[:5], 1):
            summary_content += f"{i}. {rec}\n"
        
        if executive_summary.critical_issues:
            summary_content += "\nCritical Issues:\n"
            for issue in executive_summary.critical_issues:
                summary_content += f"- {issue}\n"
        
        # Save summary report
        summary_file = self.report_storage_path / f"{report.test_session_id}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_content)


def create_report_generator(config: Optional[PerformanceTestConfig] = None) -> ReportGenerator:
    """Factory function to create report generator"""
    if config is None:
        config = PerformanceTestConfig()
    return ReportGenerator(config) 