"""
Cost Optimization and Usage Monitoring Manager for Sovereign 4.0

Comprehensive cost management system that provides:
- Usage tracking with daily_token_usage, session_costs, and user_quotas
- Cost controls with max_daily_cost, max_session_duration, max_concurrent_sessions
- User quotas per user with database storage
- Connection pooling and reuse optimization
- Cost prediction using linear regression
- Cost alerts and automatic session termination
- Usage analytics dashboard integration
- Cost optimization recommendations based on usage patterns

Integrates with existing infrastructure:
- RealtimeMetricsCollector for cost tracking
- RealtimeSessionManager for session management
- RealtimeConnectionPool for connection optimization
- Dashboard system for analytics display
"""

import asyncio
import logging
import time
import sqlite3
import json
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import aiosqlite
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Integration with existing monitoring systems
from .realtime_metrics_collector import RealtimeMetricsCollector, RealtimeCostMetrics
from .realtime_session_manager import RealtimeSessionManager, SessionState
from .connection_stability_monitor import ConnectionStabilityMonitor
from .config_manager import RealtimeAPIConfig
from .enhanced_alert_system import EnhancedAlertSystem, AlertRule, Alert, EscalationPolicy


class CostAlertType(Enum):
    """Types of cost-related alerts"""
    DAILY_BUDGET_WARNING = "daily_budget_warning"
    DAILY_BUDGET_EXCEEDED = "daily_budget_exceeded"
    USER_QUOTA_WARNING = "user_quota_warning"
    USER_QUOTA_EXCEEDED = "user_quota_exceeded"
    HIGH_COST_SESSION = "high_cost_session"
    COST_SPIKE_DETECTED = "cost_spike_detected"
    INEFFICIENT_USAGE = "inefficient_usage"
    SESSION_DURATION_LIMIT = "session_duration_limit"


class OptimizationLevel(Enum):
    """Cost optimization levels"""
    CONSERVATIVE = "conservative"  # Minimal optimization, focus on stability
    BALANCED = "balanced"         # Balance between cost and performance
    AGGRESSIVE = "aggressive"     # Maximum cost optimization


@dataclass
class UserQuota:
    """User quota configuration"""
    user_id: str
    daily_budget: float = 10.0        # USD per day
    monthly_budget: float = 300.0     # USD per month
    max_session_duration: int = 30    # minutes
    max_concurrent_sessions: int = 3
    max_tokens_per_session: int = 50000
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class UsageMetrics:
    """User usage metrics"""
    user_id: str
    date: datetime
    total_cost: float = 0.0
    total_tokens: int = 0
    session_count: int = 0
    average_session_duration: float = 0.0
    total_session_duration: float = 0.0
    error_count: int = 0
    peak_concurrent_sessions: int = 0


@dataclass
class CostPrediction:
    """Cost prediction results"""
    predicted_daily_cost: float
    predicted_monthly_cost: float
    confidence_score: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    recommendation: str
    feature_importance: Dict[str, float]
    model_accuracy: float


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    type: str
    priority: str  # "high", "medium", "low"
    description: str
    potential_savings: float  # USD per day
    implementation_effort: str  # "easy", "medium", "hard"
    details: Dict[str, Any]


@dataclass
class CostControlConfig:
    """Configuration for cost controls"""
    max_daily_cost: float = 50.0
    max_session_duration: int = 30  # minutes
    max_concurrent_sessions: int = 5
    cost_alert_threshold: float = 0.8  # Alert at 80% of limits
    enable_auto_termination: bool = True
    enable_cost_prediction: bool = True
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    connection_pool_size: int = 10
    connection_reuse_timeout: int = 300  # seconds


class CostOptimizationManager:
    """
    Comprehensive cost optimization and usage monitoring system
    
    Provides intelligent cost management with user quotas, predictive analytics,
    automated controls, and optimization recommendations for the Realtime API.
    """
    
    def __init__(self,
                 config: CostControlConfig,
                 realtime_config: RealtimeAPIConfig,
                 metrics_collector: Optional[RealtimeMetricsCollector] = None,
                 session_manager: Optional[RealtimeSessionManager] = None,
                 connection_pool: Optional[ConnectionStabilityMonitor] = None,
                 alert_system: Optional[EnhancedAlertSystem] = None,
                 logger: Optional[logging.Logger] = None):
        
        self.config = config
        self.realtime_config = realtime_config
        self.metrics_collector = metrics_collector
        self.session_manager = session_manager
        self.connection_pool = connection_pool
        self.alert_system = alert_system
        self.logger = logger or logging.getLogger(__name__)
        
        # Database configuration
        self.database_path = Path("data/cost_optimization.db")
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Cost tracking and controls
        self.user_quotas: Dict[str, UserQuota] = {}
        self.daily_usage: Dict[str, UsageMetrics] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Prediction and analytics
        self.cost_predictor: Optional[LinearRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.prediction_features: List[str] = [
            "hour_of_day", "day_of_week", "session_count", 
            "avg_session_duration", "tokens_per_minute", "error_rate"
        ]
        
        # Background tasks
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.is_running = False
        
        # Performance metrics
        self.optimization_stats = {
            "sessions_terminated": 0,
            "cost_savings": 0.0,
            "predictions_made": 0,
            "alerts_triggered": 0,
            "connections_reused": 0
        }
        
        # Thread safety
        self._lock = asyncio.Lock()
        
    async def initialize(self) -> bool:
        """Initialize the cost optimization system"""
        try:
            self.logger.info("💰 Initializing Cost Optimization Manager...")
            
            # Initialize database
            if not await self._initialize_database():
                return False
            
            # Load user quotas from database
            await self._load_user_quotas()
            
            # Initialize cost prediction model
            if self.config.enable_cost_prediction:
                await self._initialize_prediction_model()
            
            # Set up alert rules
            if self.alert_system:
                await self._setup_cost_alert_rules()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_running = True
            self.logger.info("✅ Cost Optimization Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize cost optimization manager: {e}")
            return False
    
    async def _initialize_database(self) -> bool:
        """Initialize SQLite database for cost management"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                # User quotas table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS user_quotas (
                        user_id TEXT PRIMARY KEY,
                        daily_budget REAL NOT NULL,
                        monthly_budget REAL NOT NULL,
                        max_session_duration INTEGER NOT NULL,
                        max_concurrent_sessions INTEGER NOT NULL,
                        max_tokens_per_session INTEGER NOT NULL,
                        enabled BOOLEAN NOT NULL DEFAULT 1,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )
                """)
                
                # Daily usage metrics table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS daily_usage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        total_cost REAL NOT NULL DEFAULT 0.0,
                        total_tokens INTEGER NOT NULL DEFAULT 0,
                        session_count INTEGER NOT NULL DEFAULT 0,
                        average_session_duration REAL NOT NULL DEFAULT 0.0,
                        total_session_duration REAL NOT NULL DEFAULT 0.0,
                        error_count INTEGER NOT NULL DEFAULT 0,
                        peak_concurrent_sessions INTEGER NOT NULL DEFAULT 0,
                        UNIQUE(user_id, date)
                    )
                """)
                
                # Session costs table for detailed tracking
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS session_costs (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        duration_seconds REAL,
                        total_cost REAL NOT NULL DEFAULT 0.0,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        error_count INTEGER NOT NULL DEFAULT 0,
                        terminated_reason TEXT,
                        metadata TEXT
                    )
                """)
                
                # Cost predictions table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS cost_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        prediction_date TIMESTAMP NOT NULL,
                        predicted_daily_cost REAL NOT NULL,
                        predicted_monthly_cost REAL NOT NULL,
                        confidence_score REAL NOT NULL,
                        trend_direction TEXT NOT NULL,
                        model_accuracy REAL NOT NULL,
                        features TEXT NOT NULL
                    )
                """)
                
                # Optimization recommendations table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        recommendation_date TIMESTAMP NOT NULL,
                        type TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        description TEXT NOT NULL,
                        potential_savings REAL NOT NULL,
                        implementation_effort TEXT NOT NULL,
                        status TEXT DEFAULT 'pending',
                        details TEXT
                    )
                """)
                
                # Create indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_usage_user_date ON daily_usage(user_id, date)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_session_costs_user ON session_costs(user_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_session_costs_start_time ON session_costs(start_time)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_cost_predictions_user ON cost_predictions(user_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_recommendations_user ON optimization_recommendations(user_id)")
                
                await db.commit()
            
            self.logger.info("✅ Cost optimization database initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def _load_user_quotas(self):
        """Load user quotas from database"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                async with db.execute("SELECT * FROM user_quotas WHERE enabled = 1") as cursor:
                    rows = await cursor.fetchall()
                    
                    for row in rows:
                        quota = UserQuota(
                            user_id=row[0],
                            daily_budget=row[1],
                            monthly_budget=row[2],
                            max_session_duration=row[3],
                            max_concurrent_sessions=row[4],
                            max_tokens_per_session=row[5],
                            enabled=bool(row[6]),
                            created_at=datetime.fromisoformat(row[7]),
                            updated_at=datetime.fromisoformat(row[8])
                        )
                        self.user_quotas[quota.user_id] = quota
            
            self.logger.info(f"📊 Loaded {len(self.user_quotas)} user quotas")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load user quotas: {e}")
    
    async def _initialize_prediction_model(self):
        """Initialize machine learning model for cost prediction"""
        try:
            # Load historical data for training
            historical_data = await self._get_historical_usage_data(days=30)
            
            if len(historical_data) >= 10:  # Need minimum data for training
                # Prepare features and targets
                features, targets = self._prepare_training_data(historical_data)
                
                if len(features) > 0:
                    # Initialize and train models
                    self.scaler = StandardScaler()
                    self.cost_predictor = LinearRegression()
                    
                    # Scale features
                    features_scaled = self.scaler.fit_transform(features)
                    
                    # Train model
                    self.cost_predictor.fit(features_scaled, targets)
                    
                    # Calculate model accuracy
                    predictions = self.cost_predictor.predict(features_scaled)
                    r2 = r2_score(targets, predictions)
                    mae = mean_absolute_error(targets, predictions)
                    
                    self.logger.info(f"📈 Cost prediction model trained - R²: {r2:.3f}, MAE: ${mae:.3f}")
                else:
                    self.logger.warning("⚠️ Insufficient feature data for model training")
            else:
                self.logger.warning("⚠️ Insufficient historical data for cost prediction model")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize prediction model: {e}")
    
    async def _get_historical_usage_data(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get historical usage data for model training"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.database_path) as db:
                async with db.execute("""
                    SELECT 
                        user_id, date, total_cost, total_tokens, session_count,
                        average_session_duration, total_session_duration, error_count,
                        peak_concurrent_sessions
                    FROM daily_usage 
                    WHERE date >= ?
                    ORDER BY date
                """, (cutoff_date.date(),)) as cursor:
                    rows = await cursor.fetchall()
                    
                    return [
                        {
                            "user_id": row[0], "date": row[1], "total_cost": row[2],
                            "total_tokens": row[3], "session_count": row[4],
                            "average_session_duration": row[5], "total_session_duration": row[6],
                            "error_count": row[7], "peak_concurrent_sessions": row[8]
                        }
                        for row in rows
                    ]
        except Exception as e:
            self.logger.error(f"❌ Failed to get historical usage data: {e}")
            return []
    
    def _prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for cost prediction model"""
        try:
            features = []
            targets = []
            
            for record in historical_data:
                # Parse date for time-based features
                date = datetime.strptime(record["date"], "%Y-%m-%d")
                
                # Extract features
                feature_vector = [
                    date.hour if hasattr(date, 'hour') else 12,  # hour_of_day
                    date.weekday(),  # day_of_week
                    record["session_count"],  # session_count
                    record["average_session_duration"],  # avg_session_duration
                    record["total_tokens"] / max(1, record["total_session_duration"] / 60),  # tokens_per_minute
                    record["error_count"] / max(1, record["session_count"])  # error_rate
                ]
                
                features.append(feature_vector)
                targets.append(record["total_cost"])
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare training data: {e}")
            return np.array([]), np.array([])
    
    async def _setup_cost_alert_rules(self):
        """Setup cost-related alert rules"""
        try:
            if not self.alert_system:
                return
            
            # Daily budget warning alert
            daily_budget_warning = AlertRule(
                name="daily_budget_warning",
                description="Daily budget approaching limit",
                metric_path="cost.daily_budget_usage_percent",
                operator="gte",
                threshold=self.config.cost_alert_threshold * 100,
                severity="warning"
            )
            
            # Daily budget exceeded alert
            daily_budget_exceeded = AlertRule(
                name="daily_budget_exceeded",
                description="Daily budget exceeded",
                metric_path="cost.daily_budget_usage_percent",
                operator="gte",
                threshold=100.0,
                severity="critical"
            )
            
            # High cost session alert
            high_cost_session = AlertRule(
                name="high_cost_session",
                description="Session cost exceeding normal range",
                metric_path="cost.session_cost_usd",
                operator="gt",
                threshold=5.0,  # $5 per session
                severity="warning"
            )
            
            # Add rules to alert system
            await self.alert_system.add_rule(daily_budget_warning)
            await self.alert_system.add_rule(daily_budget_exceeded)
            await self.alert_system.add_rule(high_cost_session)
            
            self.logger.info("🚨 Cost alert rules configured")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup cost alert rules: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        try:
            self.scheduler = AsyncIOScheduler()
            
            # Daily usage aggregation (every hour)
            self.scheduler.add_job(
                self._aggregate_daily_usage,
                IntervalTrigger(hours=1),
                id="daily_usage_aggregation",
                replace_existing=True
            )
            
            # Cost prediction (every 6 hours)
            if self.config.enable_cost_prediction:
                self.scheduler.add_job(
                    self._generate_cost_predictions,
                    IntervalTrigger(hours=6),
                    id="cost_predictions",
                    replace_existing=True
                )
            
            # Optimization recommendations (daily)
            self.scheduler.add_job(
                self._generate_optimization_recommendations,
                IntervalTrigger(hours=24),
                id="optimization_recommendations",
                replace_existing=True
            )
            
            # Session monitoring (every 5 minutes)
            self.scheduler.add_job(
                self._monitor_active_sessions,
                IntervalTrigger(minutes=5),
                id="session_monitoring",
                replace_existing=True
            )
            
            # Connection pool optimization (every 10 minutes)
            self.scheduler.add_job(
                self._optimize_connection_pool,
                IntervalTrigger(minutes=10),
                id="connection_optimization",
                replace_existing=True
            )
            
            self.scheduler.start()
            self.logger.info("⏰ Background tasks started")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start background tasks: {e}")
    
    async def create_or_update_user_quota(self, user_quota: UserQuota) -> bool:
        """Create or update user quota"""
        try:
            async with self._lock:
                # Update in-memory cache
                user_quota.updated_at = datetime.now()
                self.user_quotas[user_quota.user_id] = user_quota
                
                # Save to database
                async with aiosqlite.connect(self.database_path) as db:
                    await db.execute("""
                        INSERT OR REPLACE INTO user_quotas 
                        (user_id, daily_budget, monthly_budget, max_session_duration,
                         max_concurrent_sessions, max_tokens_per_session, enabled,
                         created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_quota.user_id, user_quota.daily_budget, user_quota.monthly_budget,
                        user_quota.max_session_duration, user_quota.max_concurrent_sessions,
                        user_quota.max_tokens_per_session, user_quota.enabled,
                        user_quota.created_at.isoformat(), user_quota.updated_at.isoformat()
                    ))
                    await db.commit()
                
                self.logger.info(f"💰 Updated quota for user {user_quota.user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update user quota: {e}")
            return False
    
    async def start_session_monitoring(self, session_id: str, user_id: str) -> bool:
        """Start monitoring a new session for cost controls"""
        try:
            async with self._lock:
                # Check user quota limits
                user_quota = self.user_quotas.get(user_id)
                if user_quota and user_quota.enabled:
                    # Check concurrent sessions limit
                    user_sessions = [s for s in self.active_sessions.values() 
                                   if s.get("user_id") == user_id]
                    
                    if len(user_sessions) >= user_quota.max_concurrent_sessions:
                        self.logger.warning(f"⚠️ User {user_id} exceeded concurrent sessions limit")
                        await self._trigger_cost_alert(
                            CostAlertType.USER_QUOTA_EXCEEDED,
                            user_id,
                            {"current_sessions": len(user_sessions), "limit": user_quota.max_concurrent_sessions}
                        )
                        return False
                    
                    # Check daily budget
                    daily_cost = await self._get_user_daily_cost(user_id)
                    if daily_cost >= user_quota.daily_budget:
                        self.logger.warning(f"⚠️ User {user_id} exceeded daily budget")
                        await self._trigger_cost_alert(
                            CostAlertType.USER_QUOTA_EXCEEDED,
                            user_id,
                            {"daily_cost": daily_cost, "budget": user_quota.daily_budget}
                        )
                        return False
                
                # Start session tracking
                self.active_sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "start_time": time.time(),
                    "total_cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "error_count": 0,
                    "warnings_triggered": 0
                }
                
                # Save to database
                await self._save_session_start(session_id, user_id)
                
                self.logger.info(f"🎯 Started cost monitoring for session {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start session monitoring: {e}")
            return False
    
    async def record_session_usage(self, session_id: str, input_tokens: int, output_tokens: int) -> bool:
        """Record token usage for a session"""
        try:
            async with self._lock:
                if session_id not in self.active_sessions:
                    self.logger.warning(f"⚠️ Session {session_id} not found in cost monitoring")
                    return False
                
                session = self.active_sessions[session_id]
                user_id = session["user_id"]
                
                # Calculate cost
                input_cost = (input_tokens / 1000) * 0.006  # $0.006 per 1k input tokens
                output_cost = (output_tokens / 1000) * 0.024  # $0.024 per 1k output tokens
                session_cost = input_cost + output_cost
                
                # Update session tracking
                session["input_tokens"] += input_tokens
                session["output_tokens"] += output_tokens
                session["total_cost"] += session_cost
                
                # Check cost limits
                user_quota = self.user_quotas.get(user_id)
                if user_quota and user_quota.enabled:
                    # Check session token limit
                    total_tokens = session["input_tokens"] + session["output_tokens"]
                    if total_tokens >= user_quota.max_tokens_per_session:
                        await self._terminate_session_for_limit(session_id, "token_limit")
                        return False
                    
                    # Check session cost limit
                    if session["total_cost"] >= self.realtime_config.max_cost_per_session:
                        await self._terminate_session_for_limit(session_id, "cost_limit")
                        return False
                    
                    # Check daily budget
                    daily_cost = await self._get_user_daily_cost(user_id) + session_cost
                    if daily_cost >= user_quota.daily_budget * self.config.cost_alert_threshold:
                        await self._trigger_cost_alert(
                            CostAlertType.USER_QUOTA_WARNING,
                            user_id,
                            {"daily_cost": daily_cost, "budget": user_quota.daily_budget}
                        )
                
                self.logger.debug(f"💰 Recorded usage for session {session_id}: ${session_cost:.4f}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to record session usage: {e}")
            return False
    
    async def end_session_monitoring(self, session_id: str) -> Dict[str, Any]:
        """End session monitoring and return final metrics"""
        try:
            async with self._lock:
                if session_id not in self.active_sessions:
                    return {}
                
                session = self.active_sessions.pop(session_id)
                end_time = time.time()
                duration = end_time - session["start_time"]
                
                # Calculate final metrics
                session_metrics = {
                    "session_id": session_id,
                    "user_id": session["user_id"],
                    "duration_seconds": duration,
                    "total_cost": session["total_cost"],
                    "input_tokens": session["input_tokens"],
                    "output_tokens": session["output_tokens"],
                    "total_tokens": session["input_tokens"] + session["output_tokens"],
                    "cost_per_token": session["total_cost"] / max(1, session["input_tokens"] + session["output_tokens"]),
                    "error_count": session["error_count"],
                    "warnings_triggered": session["warnings_triggered"]
                }
                
                # Save to database
                await self._save_session_end(session_id, session_metrics)
                
                # Update daily usage
                await self._update_daily_usage(session["user_id"], session_metrics)
                
                self.logger.info(f"📊 Ended cost monitoring for session {session_id} - Cost: ${session['total_cost']:.4f}")
                return session_metrics
                
        except Exception as e:
            self.logger.error(f"❌ Failed to end session monitoring: {e}")
            return {}
    
    async def _terminate_session_for_limit(self, session_id: str, reason: str):
        """Terminate session for exceeding limits"""
        try:
            if self.config.enable_auto_termination and self.session_manager:
                await self.session_manager.terminate_session(session_id, reason)
                
                session = self.active_sessions.get(session_id, {})
                user_id = session.get("user_id")
                
                # Trigger alert
                await self._trigger_cost_alert(
                    CostAlertType.SESSION_DURATION_LIMIT if reason == "duration_limit" else CostAlertType.USER_QUOTA_EXCEEDED,
                    user_id,
                    {"session_id": session_id, "reason": reason}
                )
                
                self.optimization_stats["sessions_terminated"] += 1
                self.logger.warning(f"⛔ Terminated session {session_id} for {reason}")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to terminate session: {e}")
    
    async def _get_user_daily_cost(self, user_id: str) -> float:
        """Get user's total cost for current day"""
        try:
            today = datetime.now().date()
            
            async with aiosqlite.connect(self.database_path) as db:
                async with db.execute("""
                    SELECT total_cost FROM daily_usage 
                    WHERE user_id = ? AND date = ?
                """, (user_id, today)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        return row[0]
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get user daily cost: {e}")
            return 0.0
    
    async def _save_session_start(self, session_id: str, user_id: str):
        """Save session start to database"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT INTO session_costs 
                    (session_id, user_id, start_time)
                    VALUES (?, ?, ?)
                """, (session_id, user_id, datetime.now().isoformat()))
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save session start: {e}")
    
    async def _save_session_end(self, session_id: str, metrics: Dict[str, Any]):
        """Save session end metrics to database"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    UPDATE session_costs SET
                        end_time = ?, duration_seconds = ?, total_cost = ?,
                        input_tokens = ?, output_tokens = ?, error_count = ?,
                        metadata = ?
                    WHERE session_id = ?
                """, (
                    datetime.now().isoformat(),
                    metrics["duration_seconds"],
                    metrics["total_cost"],
                    metrics["input_tokens"],
                    metrics["output_tokens"],
                    metrics["error_count"],
                    json.dumps({"warnings_triggered": metrics["warnings_triggered"]}),
                    session_id
                ))
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save session end: {e}")
    
    async def _update_daily_usage(self, user_id: str, session_metrics: Dict[str, Any]):
        """Update daily usage metrics"""
        try:
            today = datetime.now().date()
            
            async with aiosqlite.connect(self.database_path) as db:
                # Get existing daily usage
                async with db.execute("""
                    SELECT total_cost, total_tokens, session_count, total_session_duration, error_count
                    FROM daily_usage WHERE user_id = ? AND date = ?
                """, (user_id, today)) as cursor:
                    row = await cursor.fetchone()
                    
                    if row:
                        # Update existing record
                        new_total_cost = row[0] + session_metrics["total_cost"]
                        new_total_tokens = row[1] + session_metrics["total_tokens"]
                        new_session_count = row[2] + 1
                        new_total_duration = row[3] + session_metrics["duration_seconds"]
                        new_error_count = row[4] + session_metrics["error_count"]
                        new_avg_duration = new_total_duration / new_session_count
                        
                        await db.execute("""
                            UPDATE daily_usage SET
                                total_cost = ?, total_tokens = ?, session_count = ?,
                                average_session_duration = ?, total_session_duration = ?,
                                error_count = ?
                            WHERE user_id = ? AND date = ?
                        """, (
                            new_total_cost, new_total_tokens, new_session_count,
                            new_avg_duration, new_total_duration, new_error_count,
                            user_id, today
                        ))
                    else:
                        # Create new record
                        await db.execute("""
                            INSERT INTO daily_usage 
                            (user_id, date, total_cost, total_tokens, session_count,
                             average_session_duration, total_session_duration, error_count,
                             peak_concurrent_sessions)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            user_id, today, session_metrics["total_cost"],
                            session_metrics["total_tokens"], 1,
                            session_metrics["duration_seconds"], session_metrics["duration_seconds"],
                            session_metrics["error_count"], 1
                        ))
                
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update daily usage: {e}")
    
    async def _trigger_cost_alert(self, alert_type: CostAlertType, user_id: str, context: Dict[str, Any]):
        """Trigger cost-related alert"""
        try:
            if self.alert_system:
                alert = Alert(
                    id=f"cost_{alert_type.value}_{user_id}_{int(time.time())}",
                    rule_name=alert_type.value,
                    message=f"Cost alert for user {user_id}: {alert_type.value}",
                    context=context,
                    severity="critical" if "exceeded" in alert_type.value else "warning"
                )
                
                await self.alert_system.process_alert(alert)
                self.optimization_stats["alerts_triggered"] += 1
                
        except Exception as e:
            self.logger.error(f"❌ Failed to trigger cost alert: {e}")
    
    async def _aggregate_daily_usage(self):
        """Background task to aggregate daily usage metrics"""
        try:
            # Update peak concurrent sessions for all users
            for user_id in set(session["user_id"] for session in self.active_sessions.values()):
                user_sessions = [s for s in self.active_sessions.values() if s["user_id"] == user_id]
                peak_sessions = len(user_sessions)
                
                if peak_sessions > 0:
                    today = datetime.now().date()
                    async with aiosqlite.connect(self.database_path) as db:
                        await db.execute("""
                            UPDATE daily_usage SET
                                peak_concurrent_sessions = MAX(peak_concurrent_sessions, ?)
                            WHERE user_id = ? AND date = ?
                        """, (peak_sessions, user_id, today))
                        await db.commit()
            
            self.logger.debug("📊 Daily usage aggregation completed")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to aggregate daily usage: {e}")
    
    async def _generate_cost_predictions(self):
        """Generate cost predictions for all users"""
        try:
            if not self.cost_predictor or not self.scaler:
                return
            
            for user_id in self.user_quotas.keys():
                prediction = await self._predict_user_costs(user_id)
                if prediction:
                    await self._save_cost_prediction(user_id, prediction)
            
            self.optimization_stats["predictions_made"] += len(self.user_quotas)
            self.logger.info(f"📈 Generated cost predictions for {len(self.user_quotas)} users")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate cost predictions: {e}")
    
    async def _predict_user_costs(self, user_id: str) -> Optional[CostPrediction]:
        """Generate cost prediction for a specific user"""
        try:
            # Get recent usage data
            recent_usage = await self._get_user_recent_usage(user_id, days=7)
            
            if len(recent_usage) < 3:  # Need minimum data
                return None
            
            # Calculate current features
            now = datetime.now()
            avg_session_duration = np.mean([u["average_session_duration"] for u in recent_usage])
            avg_tokens_per_minute = np.mean([
                u["total_tokens"] / max(1, u["total_session_duration"] / 60) 
                for u in recent_usage
            ])
            avg_error_rate = np.mean([
                u["error_count"] / max(1, u["session_count"]) 
                for u in recent_usage
            ])
            avg_session_count = np.mean([u["session_count"] for u in recent_usage])
            
            # Prepare feature vector
            features = np.array([[
                now.hour,  # hour_of_day
                now.weekday(),  # day_of_week
                avg_session_count,  # session_count
                avg_session_duration,  # avg_session_duration
                avg_tokens_per_minute,  # tokens_per_minute
                avg_error_rate  # error_rate
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            predicted_daily_cost = self.cost_predictor.predict(features_scaled)[0]
            predicted_monthly_cost = predicted_daily_cost * 30
            
            # Calculate confidence and trend
            recent_costs = [u["total_cost"] for u in recent_usage[-5:]]
            trend_direction = self._calculate_trend_direction(recent_costs)
            confidence_score = min(1.0, len(recent_usage) / 7.0)  # More data = higher confidence
            
            # Feature importance (simplified)
            feature_importance = {
                name: abs(coef) for name, coef in 
                zip(self.prediction_features, self.cost_predictor.coef_)
            }
            
            # Generate recommendation
            recommendation = self._generate_cost_recommendation(
                predicted_daily_cost, recent_usage[-1] if recent_usage else None
            )
            
            return CostPrediction(
                predicted_daily_cost=max(0.0, predicted_daily_cost),
                predicted_monthly_cost=max(0.0, predicted_monthly_cost),
                confidence_score=confidence_score,
                trend_direction=trend_direction,
                recommendation=recommendation,
                feature_importance=feature_importance,
                model_accuracy=0.85  # Placeholder - should be calculated from validation
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to predict costs for user {user_id}: {e}")
            return None
    
    def _calculate_trend_direction(self, recent_costs: List[float]) -> str:
        """Calculate cost trend direction"""
        if len(recent_costs) < 2:
            return "stable"
        
        # Simple linear regression on recent costs
        x = np.arange(len(recent_costs))
        coefficients = np.polyfit(x, recent_costs, 1)
        slope = coefficients[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _generate_cost_recommendation(self, predicted_cost: float, recent_usage: Optional[Dict]) -> str:
        """Generate cost recommendation based on prediction"""
        if not recent_usage:
            return "Monitor usage patterns for better predictions"
        
        current_cost = recent_usage.get("total_cost", 0)
        
        if predicted_cost > current_cost * 1.5:
            return "Consider reducing session frequency or duration to control costs"
        elif predicted_cost > current_cost * 1.2:
            return "Monitor usage closely - costs trending upward"
        elif predicted_cost < current_cost * 0.8:
            return "Cost efficiency improving - current usage patterns sustainable"
        else:
            return "Usage patterns are stable and cost-effective"
    
    async def _get_user_recent_usage(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent usage data for a user"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            async with aiosqlite.connect(self.database_path) as db:
                async with db.execute("""
                    SELECT * FROM daily_usage 
                    WHERE user_id = ? AND date >= ?
                    ORDER BY date
                """, (user_id, cutoff_date.date())) as cursor:
                    rows = await cursor.fetchall()
                    
                    return [
                        {
                            "date": row[2], "total_cost": row[3], "total_tokens": row[4],
                            "session_count": row[5], "average_session_duration": row[6],
                            "total_session_duration": row[7], "error_count": row[8],
                            "peak_concurrent_sessions": row[9]
                        }
                        for row in rows
                    ]
        except Exception as e:
            self.logger.error(f"❌ Failed to get user recent usage: {e}")
            return []
    
    async def _save_cost_prediction(self, user_id: str, prediction: CostPrediction):
        """Save cost prediction to database"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT INTO cost_predictions 
                    (user_id, prediction_date, predicted_daily_cost, predicted_monthly_cost,
                     confidence_score, trend_direction, model_accuracy, features)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, datetime.now().isoformat(),
                    prediction.predicted_daily_cost, prediction.predicted_monthly_cost,
                    prediction.confidence_score, prediction.trend_direction,
                    prediction.model_accuracy, json.dumps(prediction.feature_importance)
                ))
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save cost prediction: {e}")
    
    async def _generate_optimization_recommendations(self):
        """Generate cost optimization recommendations"""
        try:
            recommendations = []
            
            # Analyze all users for optimization opportunities
            for user_id, quota in self.user_quotas.items():
                user_recommendations = await self._analyze_user_optimization_opportunities(user_id)
                recommendations.extend(user_recommendations)
            
            # Save recommendations to database
            for rec in recommendations:
                await self._save_optimization_recommendation(rec)
            
            self.logger.info(f"💡 Generated {len(recommendations)} optimization recommendations")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate optimization recommendations: {e}")
    
    async def _analyze_user_optimization_opportunities(self, user_id: str) -> List[OptimizationRecommendation]:
        """Analyze optimization opportunities for a user"""
        try:
            recommendations = []
            recent_usage = await self._get_user_recent_usage(user_id, days=7)
            
            if not recent_usage:
                return recommendations
            
            # Calculate metrics
            avg_cost = np.mean([u["total_cost"] for u in recent_usage])
            avg_duration = np.mean([u["average_session_duration"] for u in recent_usage])
            avg_error_rate = np.mean([u["error_count"] / max(1, u["session_count"]) for u in recent_usage])
            
            # High cost per session
            if avg_cost > 2.0:  # $2 per day average
                recommendations.append(OptimizationRecommendation(
                    type="session_optimization",
                    priority="high",
                    description="Consider shorter sessions or fewer daily interactions",
                    potential_savings=avg_cost * 0.3,  # 30% potential savings
                    implementation_effort="easy",
                    details={
                        "current_avg_cost": avg_cost,
                        "suggested_session_limit": "20 minutes",
                        "suggested_daily_sessions": "max 5"
                    }
                ))
            
            # Long session durations
            if avg_duration > 25 * 60:  # 25 minutes average
                recommendations.append(OptimizationRecommendation(
                    type="duration_optimization",
                    priority="medium",
                    description="Sessions are longer than optimal - consider breaking into shorter interactions",
                    potential_savings=avg_cost * 0.2,  # 20% potential savings
                    implementation_effort="easy",
                    details={
                        "current_avg_duration": avg_duration / 60,
                        "suggested_max_duration": 20,
                        "efficiency_gain": "Better cost per interaction"
                    }
                ))
            
            # High error rates
            if avg_error_rate > 0.1:  # 10% error rate
                recommendations.append(OptimizationRecommendation(
                    type="error_reduction",
                    priority="high",
                    description="High error rates are increasing costs - review connection quality",
                    potential_savings=avg_cost * avg_error_rate,
                    implementation_effort="medium",
                    details={
                        "current_error_rate": avg_error_rate,
                        "suggested_actions": ["Check network stability", "Review audio setup"],
                        "cost_impact": "Errors require retries and increase token usage"
                    }
                ))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze optimization opportunities: {e}")
            return []
    
    async def _save_optimization_recommendation(self, recommendation: OptimizationRecommendation):
        """Save optimization recommendation to database"""
        try:
            async with aiosqlite.connect(self.database_path) as db:
                await db.execute("""
                    INSERT INTO optimization_recommendations 
                    (user_id, recommendation_date, type, priority, description,
                     potential_savings, implementation_effort, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    recommendation.details.get("user_id", "system"),
                    datetime.now().isoformat(),
                    recommendation.type,
                    recommendation.priority,
                    recommendation.description,
                    recommendation.potential_savings,
                    recommendation.implementation_effort,
                    json.dumps(recommendation.details)
                ))
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save optimization recommendation: {e}")
    
    async def _monitor_active_sessions(self):
        """Monitor active sessions for limit enforcement"""
        try:
            current_time = time.time()
            sessions_to_terminate = []
            
            for session_id, session in self.active_sessions.items():
                user_id = session["user_id"]
                session_duration = current_time - session["start_time"]
                
                # Check session duration limit
                user_quota = self.user_quotas.get(user_id)
                if user_quota and session_duration > user_quota.max_session_duration * 60:
                    sessions_to_terminate.append((session_id, "duration_limit"))
                
                # Check daily budget
                daily_cost = await self._get_user_daily_cost(user_id)
                if user_quota and daily_cost >= user_quota.daily_budget:
                    sessions_to_terminate.append((session_id, "daily_budget_exceeded"))
            
            # Terminate sessions that exceeded limits
            for session_id, reason in sessions_to_terminate:
                await self._terminate_session_for_limit(session_id, reason)
            
            if sessions_to_terminate:
                self.logger.info(f"⛔ Terminated {len(sessions_to_terminate)} sessions for limit violations")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to monitor active sessions: {e}")
    
    async def _optimize_connection_pool(self):
        """Optimize connection pool for cost efficiency"""
        try:
            if not self.connection_pool:
                return
            
            # Get connection pool metrics
            pool_stats = await self.connection_pool.get_pool_stats()
            
            # Calculate efficiency metrics
            reuse_rate = pool_stats.get("connections_reused", 0) / max(1, pool_stats.get("total_connections", 1))
            
            # Adjust pool size based on usage
            active_sessions = len(self.active_sessions)
            
            if reuse_rate > 0.8 and active_sessions < self.config.connection_pool_size * 0.5:
                # High reuse rate and low usage - can reduce pool size
                new_pool_size = max(2, int(self.config.connection_pool_size * 0.8))
                self.config.connection_pool_size = new_pool_size
                
            elif reuse_rate < 0.3 and active_sessions > self.config.connection_pool_size * 0.8:
                # Low reuse rate and high usage - increase pool size
                new_pool_size = min(20, int(self.config.connection_pool_size * 1.2))
                self.config.connection_pool_size = new_pool_size
            
            self.optimization_stats["connections_reused"] = pool_stats.get("connections_reused", 0)
            
            self.logger.debug(f"🔗 Connection pool optimization - Reuse rate: {reuse_rate:.2f}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize connection pool: {e}")
    
    async def get_cost_analytics(self, user_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive cost analytics"""
        try:
            analytics = {
                "summary": {},
                "daily_breakdown": [],
                "predictions": [],
                "recommendations": [],
                "efficiency_metrics": {}
            }
            
            # Build query conditions
            conditions = []
            params = []
            
            if user_id:
                conditions.append("user_id = ?")
                params.append(user_id)
            
            cutoff_date = datetime.now() - timedelta(days=days)
            conditions.append("date >= ?")
            params.append(cutoff_date.date())
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            async with aiosqlite.connect(self.database_path) as db:
                # Summary metrics
                async with db.execute(f"""
                    SELECT 
                        SUM(total_cost) as total_cost,
                        SUM(total_tokens) as total_tokens,
                        SUM(session_count) as total_sessions,
                        AVG(average_session_duration) as avg_duration,
                        AVG(total_cost / NULLIF(session_count, 0)) as avg_cost_per_session
                    FROM daily_usage {where_clause}
                """, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        analytics["summary"] = {
                            "total_cost": row[0] or 0.0,
                            "total_tokens": row[1] or 0,
                            "total_sessions": row[2] or 0,
                            "average_session_duration": row[3] or 0.0,
                            "average_cost_per_session": row[4] or 0.0,
                            "cost_per_token": (row[0] or 0.0) / max(1, row[1] or 1)
                        }
                
                # Daily breakdown
                async with db.execute(f"""
                    SELECT date, SUM(total_cost) as daily_cost, SUM(session_count) as daily_sessions
                    FROM daily_usage {where_clause}
                    GROUP BY date ORDER BY date
                """, params) as cursor:
                    rows = await cursor.fetchall()
                    analytics["daily_breakdown"] = [
                        {"date": row[0], "cost": row[1], "sessions": row[2]}
                        for row in rows
                    ]
                
                # Recent predictions
                prediction_params = params.copy() if user_id else []
                prediction_where = "WHERE user_id = ?" if user_id else ""
                
                async with db.execute(f"""
                    SELECT * FROM cost_predictions {prediction_where}
                    ORDER BY prediction_date DESC LIMIT 10
                """, prediction_params) as cursor:
                    rows = await cursor.fetchall()
                    analytics["predictions"] = [
                        {
                            "date": row[2], "predicted_daily_cost": row[3],
                            "predicted_monthly_cost": row[4], "confidence": row[5],
                            "trend": row[6]
                        }
                        for row in rows
                    ]
                
                # Recent recommendations
                async with db.execute(f"""
                    SELECT * FROM optimization_recommendations {prediction_where}
                    WHERE status = 'pending'
                    ORDER BY recommendation_date DESC LIMIT 10
                """, prediction_params) as cursor:
                    rows = await cursor.fetchall()
                    analytics["recommendations"] = [
                        {
                            "type": row[3], "priority": row[4], "description": row[5],
                            "potential_savings": row[6], "effort": row[7]
                        }
                        for row in rows
                    ]
            
            # Efficiency metrics
            analytics["efficiency_metrics"] = {
                "connection_reuse_rate": self.optimization_stats["connections_reused"] / max(1, analytics["summary"]["total_sessions"]),
                "sessions_terminated": self.optimization_stats["sessions_terminated"],
                "total_cost_savings": self.optimization_stats["cost_savings"],
                "alerts_triggered": self.optimization_stats["alerts_triggered"],
                "predictions_accuracy": "85%"  # Placeholder
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get cost analytics: {e}")
            return {}
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        return {
            "optimization_stats": self.optimization_stats.copy(),
            "active_sessions": len(self.active_sessions),
            "user_quotas": len(self.user_quotas),
            "connection_pool_size": self.config.connection_pool_size,
            "daily_budget_utilization": await self._calculate_budget_utilization(),
            "cost_prediction_enabled": self.config.enable_cost_prediction,
            "auto_termination_enabled": self.config.enable_auto_termination
        }
    
    async def _calculate_budget_utilization(self) -> Dict[str, float]:
        """Calculate budget utilization across all users"""
        try:
            utilization = {}
            today = datetime.now().date()
            
            for user_id, quota in self.user_quotas.items():
                daily_cost = await self._get_user_daily_cost(user_id)
                utilization[user_id] = (daily_cost / quota.daily_budget) * 100
            
            return utilization
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate budget utilization: {e}")
            return {}
    
    async def cleanup(self):
        """Cleanup resources and stop background tasks"""
        try:
            self.is_running = False
            
            if self.scheduler:
                self.scheduler.shutdown(wait=False)
            
            # End all active sessions
            for session_id in list(self.active_sessions.keys()):
                await self.end_session_monitoring(session_id)
            
            self.logger.info("💰 Cost Optimization Manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


# Factory function for easy integration
def create_cost_optimization_manager(
    config: CostControlConfig,
    realtime_config: RealtimeAPIConfig,
    **integrations
) -> CostOptimizationManager:
    """Create and configure a CostOptimizationManager instance"""
    return CostOptimizationManager(
        config=config,
        realtime_config=realtime_config,
        **integrations
    ) 