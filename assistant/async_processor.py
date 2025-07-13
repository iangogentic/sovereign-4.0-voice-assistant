"""
Advanced Async Processing System for LLM Router

This module provides sophisticated async processing capabilities including:
- Progressive status updates during long-running queries
- Timeout handling and graceful degradation
- Background task management with cancellation
- Progress tracking and user feedback
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ProcessingState(Enum):
    """States for async processing tasks"""
    QUEUED = "queued"
    INITIALIZING = "initializing" 
    PROCESSING = "processing"
    PROGRESS_UPDATE = "progress_update"
    TIMEOUT_WARNING = "timeout_warning"
    COMPLETING = "completing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass
class ProcessingConfig:
    """Configuration for async processing behavior"""
    # Timeout thresholds (seconds)
    fast_timeout: float = 5.0
    deep_timeout: float = 30.0
    max_timeout: float = 60.0
    
    # Status update intervals (seconds)
    initial_status_delay: float = 2.0
    progress_update_interval: float = 5.0
    timeout_warning_threshold: float = 0.8  # 80% of timeout
    
    # Degradation settings
    enable_degradation: bool = True
    degradation_threshold: float = 0.9  # 90% of timeout
    max_retries: int = 2
    
    # Concurrency limits
    max_concurrent_deep: int = 3
    max_concurrent_fast: int = 10
    
    # Status messages
    status_messages: Dict[str, List[str]] = field(default_factory=lambda: {
        'initializing': [
            "Let me think about that...",
            "Processing your request...",
            "Looking into this for you..."
        ],
        'progress': [
            "Still working on this...",
            "Making progress...",
            "Almost there...",
            "Gathering more information...",
            "Analyzing the details..."
        ],
        'timeout_warning': [
            "This is taking longer than expected...",
            "Still processing, please wait...",
            "Working on a detailed response..."
        ],
        'degradation': [
            "Let me try a faster approach...",
            "Switching to a quicker method...",
            "Simplifying the analysis..."
        ]
    })


@dataclass
class ProcessingTask:
    """Represents an async processing task"""
    task_id: str
    query: str
    model_type: str  # 'fast' or 'deep'
    start_time: float
    timeout: float
    state: ProcessingState = ProcessingState.QUEUED
    progress: float = 0.0
    last_update: float = 0.0
    retries: int = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    cancellation_token: Optional[asyncio.Event] = field(default_factory=asyncio.Event)
    
    def __post_init__(self):
        if self.cancellation_token is None:
            self.cancellation_token = asyncio.Event()
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed processing time"""
        return time.time() - self.start_time
    
    @property
    def remaining_time(self) -> float:
        """Get remaining time before timeout"""
        return max(0, self.timeout - self.elapsed_time)
    
    @property
    def timeout_percentage(self) -> float:
        """Get percentage of timeout elapsed"""
        return min(1.0, self.elapsed_time / self.timeout)
    
    def is_cancelled(self) -> bool:
        """Check if task is cancelled"""
        return self.cancellation_token.is_set()
    
    def cancel(self):
        """Cancel the task"""
        self.state = ProcessingState.CANCELLED
        self.cancellation_token.set()


class AsyncProcessor:
    """
    Advanced async processor for LLM queries with status updates and timeout handling
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.active_tasks: Dict[str, ProcessingTask] = {}
        self.status_update_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        self.processing_semaphores = {
            'fast': asyncio.Semaphore(self.config.max_concurrent_fast),
            'deep': asyncio.Semaphore(self.config.max_concurrent_deep)
        }
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'cancelled_tasks': 0,
            'failed_tasks': 0,
            'degraded_tasks': 0,
            'avg_processing_time': 0.0,
            'timeout_rate': 0.0
        }
        
        logger.info("AsyncProcessor initialized with advanced timeout and degradation handling")
    
    def set_status_update_callback(self, callback: Callable[[str, str], Awaitable[None]]):
        """Set callback function for sending status updates"""
        self.status_update_callback = callback
    
    async def process_async(
        self,
        query: str,
        model_type: str,
        processor_func: Callable[[str], Awaitable[Any]],
        request_id: Optional[str] = None
    ) -> Any:
        """
        Process a query asynchronously with status updates and timeout handling
        
        Args:
            query: The query text to process
            model_type: 'fast' or 'deep'
            processor_func: Async function that actually processes the query
            request_id: Optional request ID for tracking
            
        Returns:
            Processing result or degraded result
        """
        # Create task
        task_id = request_id or str(uuid.uuid4())
        timeout = self.config.fast_timeout if model_type == 'fast' else self.config.deep_timeout
        
        task = ProcessingTask(
            task_id=task_id,
            query=query,
            model_type=model_type,
            start_time=time.time(),
            timeout=timeout
        )
        
        self.active_tasks[task_id] = task
        self.stats['total_tasks'] += 1
        
        try:
            # Acquire semaphore for concurrency control
            async with self.processing_semaphores[model_type]:
                # Start background status updates
                status_task = asyncio.create_task(self._manage_status_updates(task))
                
                # Start the actual processing
                processing_task = asyncio.create_task(self._execute_with_timeout(task, processor_func))
                
                try:
                    # Wait for processing to complete
                    result = await processing_task
                    
                    # Cancel status updates
                    status_task.cancel()
                    
                    # Update task state
                    task.state = ProcessingState.COMPLETED
                    task.result = result
                    task.progress = 1.0
                    
                    # Update statistics
                    self.stats['completed_tasks'] += 1
                    self._update_avg_processing_time(task.elapsed_time)
                    
                    logger.info(f"Task {task_id} completed successfully in {task.elapsed_time:.2f}s")
                    return result
                    
                except asyncio.TimeoutError:
                    # Handle timeout - try degradation
                    status_task.cancel()
                    
                    if self.config.enable_degradation and task.retries < self.config.max_retries:
                        logger.warning(f"Task {task_id} timed out, attempting degradation")
                        try:
                            result = await self._handle_degradation(task, processor_func)
                            self.stats['completed_tasks'] += 1
                            self._update_avg_processing_time(task.elapsed_time)
                            return result
                        except Exception as e:
                            # Degradation also failed
                            task.state = ProcessingState.FAILED
                            task.error = f"Processing failed even with degradation: {str(e)}"
                            self.stats['failed_tasks'] += 1
                            self._update_timeout_rate()
                            raise
                    else:
                        task.state = ProcessingState.FAILED
                        task.error = f"Processing timeout after {timeout}s"
                        self.stats['failed_tasks'] += 1
                        self._update_timeout_rate()
                        
                        raise asyncio.TimeoutError(f"Query processing timed out after {timeout}s")
                
                except asyncio.CancelledError:
                    # Handle cancellation
                    status_task.cancel()
                    task.state = ProcessingState.CANCELLED
                    self.stats['cancelled_tasks'] += 1
                    
                    logger.info(f"Task {task_id} was cancelled")
                    raise
                
                except Exception as e:
                    # Handle other errors
                    status_task.cancel()
                    task.state = ProcessingState.FAILED
                    task.error = str(e)
                    self.stats['failed_tasks'] += 1
                    
                    logger.error(f"Task {task_id} failed: {e}")
                    raise
        
        finally:
            # Cleanup
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _execute_with_timeout(self, task: ProcessingTask, processor_func: Callable[[str], Awaitable[Any]]) -> Any:
        """Execute processing function with timeout and cancellation support"""
        task.state = ProcessingState.INITIALIZING
        
        try:
            # Wait for either completion or timeout
            result = await asyncio.wait_for(
                self._cancellable_processor(task, processor_func),
                timeout=task.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {task.task_id} exceeded timeout of {task.timeout}s")
            raise
    
    async def _cancellable_processor(self, task: ProcessingTask, processor_func: Callable[[str], Awaitable[Any]]) -> Any:
        """Wrapper to make processor function cancellable"""
        task.state = ProcessingState.PROCESSING
        
        # Create the actual processing task
        processing_future = asyncio.create_task(processor_func(task.query))
        
        # Wait for either completion or cancellation
        done, pending = await asyncio.wait(
            [processing_future, asyncio.create_task(task.cancellation_token.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel any pending tasks
        for p in pending:
            p.cancel()
        
        # Check if we were cancelled
        if task.is_cancelled():
            processing_future.cancel()
            raise asyncio.CancelledError("Task was cancelled")
        
        # Return the result
        return processing_future.result()
    
    async def _manage_status_updates(self, task: ProcessingTask):
        """Manage progressive status updates for a task"""
        try:
            # Initial delay before first status update
            await asyncio.sleep(self.config.initial_status_delay)
            
            if task.is_cancelled():
                return
            
            # Send initial status
            await self._send_status_update(task, 'initializing')
            
            # Continue sending progress updates
            while task.state in [ProcessingState.INITIALIZING, ProcessingState.PROCESSING]:
                await asyncio.sleep(self.config.progress_update_interval)
                
                if task.is_cancelled():
                    break
                
                # Check for timeout warning
                if task.timeout_percentage >= self.config.timeout_warning_threshold:
                    task.state = ProcessingState.TIMEOUT_WARNING
                    await self._send_status_update(task, 'timeout_warning')
                    break
                else:
                    task.state = ProcessingState.PROGRESS_UPDATE
                    await self._send_status_update(task, 'progress')
        
        except asyncio.CancelledError:
            # Normal cancellation when processing completes
            pass
        except Exception as e:
            logger.error(f"Error in status updates for task {task.task_id}: {e}")
    
    async def _send_status_update(self, task: ProcessingTask, message_type: str):
        """Send a status update message"""
        if not self.status_update_callback:
            return
        
        # Select appropriate message
        messages = self.config.status_messages.get(message_type, ["Processing..."])
        import random
        message = random.choice(messages)
        
        # Add progress info for longer tasks
        if message_type == 'progress' and task.model_type == 'deep':
            progress_pct = int(task.timeout_percentage * 100)
            message += f" ({progress_pct}% through)"
        
        try:
            await self.status_update_callback(task.task_id, message)
            task.last_update = time.time()
            logger.debug(f"Sent status update for task {task.task_id}: {message}")
        except Exception as e:
            logger.error(f"Failed to send status update for task {task.task_id}: {e}")
    
    async def _handle_degradation(self, task: ProcessingTask, processor_func: Callable[[str], Awaitable[Any]]) -> Any:
        """Handle graceful degradation when processing times out"""
        task.state = ProcessingState.DEGRADED
        task.retries += 1
        
        # Send degradation message
        await self._send_status_update(task, 'degradation')
        
        # Try with shorter timeout and simpler processing
        degraded_timeout = min(self.config.fast_timeout, task.timeout * 0.5)
        task.timeout = degraded_timeout
        task.start_time = time.time()  # Reset timer
        
        logger.info(f"Attempting degraded processing for task {task.task_id} with {degraded_timeout}s timeout")
        
        try:
            # Try again with shorter timeout
            result = await asyncio.wait_for(
                processor_func(task.query),  # Call processor directly for degradation
                timeout=degraded_timeout
            )
            
            self.stats['degraded_tasks'] += 1
            logger.info(f"Task {task.task_id} completed with degradation")
            return result
            
        except asyncio.TimeoutError:
            # Even degraded processing failed
            task.state = ProcessingState.FAILED
            task.error = f"Processing failed even with degradation (timeout: {degraded_timeout}s)"
            
            logger.error(f"Degraded processing failed for task {task.task_id}")
            raise asyncio.TimeoutError("Processing failed even with degraded approach")
    
    def _update_avg_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        completed = self.stats['completed_tasks']
        if completed == 1:
            self.stats['avg_processing_time'] = processing_time
        else:
            # Rolling average
            current_avg = self.stats['avg_processing_time']
            self.stats['avg_processing_time'] = (current_avg * (completed - 1) + processing_time) / completed
    
    def _update_timeout_rate(self):
        """Update timeout rate statistics"""
        total = self.stats['total_tasks']
        timeouts = self.stats['failed_tasks']
        self.stats['timeout_rate'] = timeouts / total if total > 0 else 0.0
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.cancel()
            logger.info(f"Cancelled task {task_id}")
            return True
        return False
    
    def cancel_all_tasks(self):
        """Cancel all active tasks"""
        for task in self.active_tasks.values():
            task.cancel()
        logger.info(f"Cancelled {len(self.active_tasks)} active tasks")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task.task_id,
                'state': task.state.value,
                'progress': task.progress,
                'elapsed_time': task.elapsed_time,
                'remaining_time': task.remaining_time,
                'timeout_percentage': task.timeout_percentage,
                'retries': task.retries
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return {
            'active_tasks': len(self.active_tasks),
            'total_tasks': self.stats['total_tasks'],
            'completed_tasks': self.stats['completed_tasks'],
            'cancelled_tasks': self.stats['cancelled_tasks'],
            'failed_tasks': self.stats['failed_tasks'],
            'degraded_tasks': self.stats['degraded_tasks'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'timeout_rate': self.stats['timeout_rate'],
            'success_rate': self.stats['completed_tasks'] / self.stats['total_tasks'] if self.stats['total_tasks'] > 0 else 0.0
        } 