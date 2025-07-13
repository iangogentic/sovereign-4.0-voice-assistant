"""
Realtime Session Manager for OpenAI Realtime API

Provides comprehensive session management with:
- Session lifecycle management with proper states
- SQLite database persistence for session metadata and conversation history
- Concurrent session handling with asyncio
- Session timeout management and cleanup
- Network interruption recovery mechanisms
- Conversation item tracking and persistence
"""

import asyncio
import sqlite3
import uuid
import time
import logging
from typing import Optional, Dict, List, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import aiosqlite
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionState(Enum):
    """Session lifecycle states"""
    IDLE = "idle"
    CONNECTING = "connecting" 
    CONNECTED = "connected"
    ACTIVE = "active"
    DISCONNECTING = "disconnecting"
    ERROR = "error"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class ConversationItemType(Enum):
    """Types of conversation items"""
    MESSAGE = "message"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    AUDIO = "audio"
    SYSTEM = "system"


@dataclass
class SessionMetadata:
    """Session metadata structure"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionState
    model: str
    voice: str
    instructions: Optional[str] = None
    temperature: float = 0.8
    max_response_output_tokens: str = "inf"
    user_id: Optional[str] = None
    connection_attempts: int = 0
    error_count: int = 0
    recovery_attempts: int = 0


@dataclass
class ConversationItem:
    """Individual conversation item"""
    item_id: str
    session_id: str
    type: ConversationItemType
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    audio_data: Optional[bytes] = None


@dataclass
class SessionConfig:
    """Configuration for session management"""
    database_path: str = "data/realtime_sessions.db"
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 10
    max_recovery_attempts: int = 3
    cleanup_interval_minutes: int = 5
    max_session_age_hours: int = 24
    enable_persistence: bool = True
    enable_recovery: bool = True


class RealtimeSessionManager:
    """
    Comprehensive session management for OpenAI Realtime API
    
    Handles session lifecycle, persistence, timeout management,
    and recovery mechanisms for robust production usage.
    """
    
    def __init__(self, config: SessionConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Active sessions tracking
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Database connection
        self.db_path = Path(config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        # Event callbacks
        self.on_session_created: Optional[Callable] = None
        self.on_session_expired: Optional[Callable] = None
        self.on_session_recovered: Optional[Callable] = None
        self.on_session_error: Optional[Callable] = None
        
        # Performance metrics
        self.total_sessions_created = 0
        self.total_sessions_expired = 0
        self.total_recovery_attempts = 0
        self.total_recovery_successes = 0
    
    async def initialize(self) -> bool:
        """Initialize the session manager"""
        try:
            self.logger.info("🔄 Initializing Realtime Session Manager...")
            
            # Initialize database
            if not await self._initialize_database():
                return False
            
            # Load active sessions from database
            await self._load_active_sessions()
            
            # Start background cleanup task
            if self.config.cleanup_interval_minutes > 0:
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            self.is_running = True
            self.logger.info("✅ Realtime Session Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize session manager: {e}")
            return False
    
    async def _initialize_database(self) -> bool:
        """Initialize SQLite database with required tables"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Sessions table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP NOT NULL,
                        last_activity TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        model TEXT NOT NULL,
                        voice TEXT NOT NULL,
                        instructions TEXT,
                        temperature REAL DEFAULT 0.8,
                        max_response_output_tokens TEXT DEFAULT 'inf',
                        user_id TEXT,
                        connection_attempts INTEGER DEFAULT 0,
                        error_count INTEGER DEFAULT 0,
                        recovery_attempts INTEGER DEFAULT 0,
                        metadata TEXT
                    )
                """)
                
                # Conversation items table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_items (
                        item_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        type TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        metadata TEXT,
                        audio_data BLOB,
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )
                """)
                
                # Indexes for performance
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_session ON conversation_items(session_id)")
                await db.execute("CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation_items(timestamp)")
                
                await db.commit()
                
            self.logger.info("✅ Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize database: {e}")
            return False
    
    async def _load_active_sessions(self):
        """Load active sessions from database"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM sessions 
                    WHERE status IN ('connected', 'active') 
                    AND last_activity > datetime('now', '-1 hour')
                """) as cursor:
                    rows = await cursor.fetchall()
                    
                    for row in rows:
                        session = self._session_from_db_row(row)
                        self.active_sessions[session.session_id] = session
                        self.session_locks[session.session_id] = asyncio.Lock()
            
            self.logger.info(f"📚 Loaded {len(self.active_sessions)} active sessions from database")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load active sessions: {e}")
    
    def _session_from_db_row(self, row) -> SessionMetadata:
        """Convert database row to SessionMetadata"""
        return SessionMetadata(
            session_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            last_activity=datetime.fromisoformat(row[2]),
            status=SessionState(row[3]),
            model=row[4],
            voice=row[5],
            instructions=row[6],
            temperature=row[7],
            max_response_output_tokens=row[8],
            user_id=row[9],
            connection_attempts=row[10],
            error_count=row[11],
            recovery_attempts=row[12]
        )
    
    async def create_session(self, model: str, voice: str, instructions: Optional[str] = None,
                           user_id: Optional[str] = None, **kwargs) -> str:
        """Create a new session"""
        try:
            # Check concurrent session limit
            if len(self.active_sessions) >= self.config.max_concurrent_sessions:
                raise RuntimeError(f"Maximum concurrent sessions ({self.config.max_concurrent_sessions}) reached")
            
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            
            # Create session metadata
            now = datetime.now()
            session = SessionMetadata(
                session_id=session_id,
                created_at=now,
                last_activity=now,
                status=SessionState.IDLE,
                model=model,
                voice=voice,
                instructions=instructions,
                temperature=kwargs.get('temperature', 0.8),
                max_response_output_tokens=kwargs.get('max_response_output_tokens', 'inf'),
                user_id=user_id
            )
            
            # Store in database
            if self.config.enable_persistence:
                await self._save_session_to_db(session)
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            self.session_locks[session_id] = asyncio.Lock()
            
            # Trigger callback
            if self.on_session_created:
                await self.on_session_created(session_id, session)
            
            self.total_sessions_created += 1
            self.logger.info(f"✅ Created session {session_id}")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create session: {e}")
            raise
    
    async def _save_session_to_db(self, session: SessionMetadata):
        """Save session metadata to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO sessions 
                (id, created_at, last_activity, status, model, voice, instructions,
                 temperature, max_response_output_tokens, user_id, connection_attempts,
                 error_count, recovery_attempts, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at.isoformat(),
                session.last_activity.isoformat(),
                session.status.value,
                session.model,
                session.voice,
                session.instructions,
                session.temperature,
                session.max_response_output_tokens,
                session.user_id,
                session.connection_attempts,
                session.error_count,
                session.recovery_attempts,
                ""  # metadata placeholder
            ))
            await db.commit()
    
    async def update_session_state(self, session_id: str, new_state: SessionState,
                                 error_info: Optional[Dict] = None) -> bool:
        """Update session state with proper lifecycle management"""
        try:
            if session_id not in self.active_sessions:
                self.logger.warning(f"⚠️ Session {session_id} not found in active sessions")
                return False
            
            async with self.session_locks[session_id]:
                session = self.active_sessions[session_id]
                old_state = session.status
                
                # Validate state transition
                if not self._is_valid_state_transition(old_state, new_state):
                    self.logger.warning(f"⚠️ Invalid state transition: {old_state} -> {new_state}")
                    return False
                
                # Update session
                session.status = new_state
                session.last_activity = datetime.now()
                
                # Handle error states
                if new_state == SessionState.ERROR:
                    session.error_count += 1
                    if error_info:
                        self.logger.error(f"❌ Session {session_id} error: {error_info}")
                
                # Save to database
                if self.config.enable_persistence:
                    await self._save_session_to_db(session)
                
                self.logger.info(f"🔄 Session {session_id}: {old_state.value} -> {new_state.value}")
                
                # Trigger callbacks
                if new_state == SessionState.ERROR and self.on_session_error:
                    await self.on_session_error(session_id, error_info)
                
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to update session state: {e}")
            return False
    
    def _is_valid_state_transition(self, from_state: SessionState, to_state: SessionState) -> bool:
        """Validate state transitions"""
        valid_transitions = {
            SessionState.IDLE: [SessionState.CONNECTING, SessionState.ERROR, SessionState.TERMINATED],
            SessionState.CONNECTING: [SessionState.CONNECTED, SessionState.ERROR, SessionState.DISCONNECTING],
            SessionState.CONNECTED: [SessionState.ACTIVE, SessionState.ERROR, SessionState.DISCONNECTING],
            SessionState.ACTIVE: [SessionState.CONNECTED, SessionState.ERROR, SessionState.DISCONNECTING],
            SessionState.DISCONNECTING: [SessionState.IDLE, SessionState.ERROR, SessionState.TERMINATED],
            SessionState.ERROR: [SessionState.CONNECTING, SessionState.TERMINATED, SessionState.EXPIRED],
            SessionState.EXPIRED: [SessionState.TERMINATED],
            SessionState.TERMINATED: []  # Terminal state
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    async def add_conversation_item(self, session_id: str, item_type: ConversationItemType,
                                  role: str, content: str, metadata: Optional[Dict] = None,
                                  audio_data: Optional[bytes] = None) -> str:
        """Add conversation item to session"""
        try:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            # Generate item ID
            item_id = str(uuid.uuid4())
            
            # Create conversation item
            item = ConversationItem(
                item_id=item_id,
                session_id=session_id,
                type=item_type,
                role=role,
                content=content,
                timestamp=datetime.now(),
                metadata=metadata or {},
                audio_data=audio_data
            )
            
            # Save to database
            if self.config.enable_persistence:
                await self._save_conversation_item_to_db(item)
            
            # Update session activity
            await self._update_session_activity(session_id)
            
            self.logger.debug(f"💬 Added conversation item {item_id} to session {session_id}")
            
            return item_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add conversation item: {e}")
            raise
    
    async def _save_conversation_item_to_db(self, item: ConversationItem):
        """Save conversation item to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversation_items 
                (item_id, session_id, type, role, content, timestamp, metadata, audio_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.item_id,
                item.session_id,
                item.type.value,
                item.role,
                item.content,
                item.timestamp.isoformat(),
                json.dumps(item.metadata),
                item.audio_data
            ))
            await db.commit()
    
    async def _update_session_activity(self, session_id: str):
        """Update session last activity timestamp"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].last_activity = datetime.now()
    
    async def get_conversation_history(self, session_id: str, limit: int = 50) -> List[ConversationItem]:
        """Get conversation history for session"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT * FROM conversation_items 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (session_id, limit)) as cursor:
                    rows = await cursor.fetchall()
                    
                    items = []
                    for row in rows:
                        items.append(ConversationItem(
                            item_id=row[0],
                            session_id=row[1],
                            type=ConversationItemType(row[2]),
                            role=row[3],
                            content=row[4],
                            timestamp=datetime.fromisoformat(row[5]),
                            metadata=json.loads(row[6]) if row[6] else {},
                            audio_data=row[7]
                        ))
                    
                    return list(reversed(items))  # Return in chronological order
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get conversation history: {e}")
            return []
    
    async def attempt_session_recovery(self, session_id: str) -> bool:
        """Attempt to recover a failed session"""
        try:
            if not self.config.enable_recovery:
                return False
            
            if session_id not in self.active_sessions:
                return False
            
            session = self.active_sessions[session_id]
            
            # Check recovery limits
            if session.recovery_attempts >= self.config.max_recovery_attempts:
                self.logger.warning(f"⚠️ Max recovery attempts reached for session {session_id}")
                await self.terminate_session(session_id, "max_recovery_attempts")
                return False
            
            # Increment recovery attempts
            session.recovery_attempts += 1
            self.total_recovery_attempts += 1
            
            # Reset session to connecting state
            await self.update_session_state(session_id, SessionState.CONNECTING)
            
            # Trigger recovery callback
            if self.on_session_recovered:
                await self.on_session_recovered(session_id)
            
            self.total_recovery_successes += 1
            self.logger.info(f"🔄 Attempting recovery for session {session_id} (attempt {session.recovery_attempts})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to recover session: {e}")
            return False
    
    async def terminate_session(self, session_id: str, reason: str = "manual"):
        """Terminate a session and clean up resources"""
        try:
            if session_id not in self.active_sessions:
                return
            
            async with self.session_locks[session_id]:
                # Update state to terminated
                await self.update_session_state(session_id, SessionState.TERMINATED)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                del self.session_locks[session_id]
                
                self.logger.info(f"🗑️ Terminated session {session_id} (reason: {reason})")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to terminate session: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup task for expired sessions"""
        while self.is_running:
            try:
                await self._cleanup_expired_sessions()
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                
            except Exception as e:
                self.logger.error(f"❌ Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            now = datetime.now()
            timeout_threshold = now - timedelta(minutes=self.config.session_timeout_minutes)
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.last_activity < timeout_threshold:
                    expired_sessions.append(session_id)
            
            # Clean up expired sessions
            for session_id in expired_sessions:
                await self.update_session_state(session_id, SessionState.EXPIRED)
                await self.terminate_session(session_id, "timeout")
                
                if self.on_session_expired:
                    await self.on_session_expired(session_id)
                
                self.total_sessions_expired += 1
            
            if expired_sessions:
                self.logger.info(f"🧹 Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error(f"❌ Error cleaning up expired sessions: {e}")
    
    async def get_session_info(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session information"""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, SessionMetadata]:
        """Get all active sessions"""
        return self.active_sessions.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get session manager metrics"""
        return {
            "active_sessions_count": len(self.active_sessions),
            "total_sessions_created": self.total_sessions_created,
            "total_sessions_expired": self.total_sessions_expired,
            "total_recovery_attempts": self.total_recovery_attempts,
            "total_recovery_successes": self.total_recovery_successes,
            "recovery_success_rate": (
                self.total_recovery_successes / self.total_recovery_attempts 
                if self.total_recovery_attempts > 0 else 0
            ),
            "is_running": self.is_running,
            "config": {
                "max_concurrent_sessions": self.config.max_concurrent_sessions,
                "session_timeout_minutes": self.config.session_timeout_minutes,
                "max_recovery_attempts": self.config.max_recovery_attempts
            }
        }
    
    async def cleanup(self):
        """Clean up session manager resources"""
        try:
            self.is_running = False
            
            # Cancel cleanup task
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass
            
            # Terminate all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.terminate_session(session_id, "shutdown")
            
            self.logger.info("🧹 Session manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


def create_session_manager(config: Optional[SessionConfig] = None) -> RealtimeSessionManager:
    """Factory function to create RealtimeSessionManager"""
    if config is None:
        config = SessionConfig()
    
    return RealtimeSessionManager(config) 