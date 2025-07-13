import os
import re
import asyncio
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import httpx
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class KimiConfig:
    """Configuration for Kimi K2 API integration"""
    api_key: Optional[str] = None
    base_url: str = "https://api.moonshot.cn/v1"
    model_id: str = "kimi-k2"
    max_tokens: int = 4000
    temperature: float = 0.1  # Lower for more deterministic code
    timeout: float = 60.0
    max_context_files: int = 10
    max_context_size: int = 50000  # Max characters of code context
    
    # Code-specific settings
    supported_languages: List[str] = field(default_factory=lambda: [
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp',
        'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'html',
        'css', 'sql', 'bash', 'yaml', 'json', 'xml', 'markdown'
    ])
    
    # File patterns to include in context
    include_patterns: List[str] = field(default_factory=lambda: [
        '*.py', '*.js', '*.ts', '*.jsx', '*.tsx', '*.java', '*.cpp', '*.c',
        '*.cs', '*.go', '*.rs', '*.php', '*.rb', '*.swift', '*.kt', '*.scala',
        '*.html', '*.css', '*.sql', '*.sh', '*.bash', '*.yaml', '*.yml',
        '*.json', '*.xml', '*.md', '*.txt', '*.conf', '*.config'
    ])
    
    # Files to exclude from context
    exclude_patterns: List[str] = field(default_factory=lambda: [
        '*.pyc', '*.pyo', '*.pyd', '__pycache__/*', '.git/*', 'node_modules/*',
        '*.min.js', '*.min.css', '*.bundle.js', '*.log', '*.tmp', '*.temp',
        'dist/*', 'build/*', '.env', '.env.*', 'venv/*', '.venv/*'
    ])

@dataclass
class CodeContext:
    """Represents code context for a request"""
    current_file: Optional[str] = None
    current_file_content: Optional[str] = None
    current_line: Optional[int] = None
    project_files: List[Dict[str, Any]] = field(default_factory=list)
    git_status: Optional[str] = None
    error_content: Optional[str] = None  # From OCR system
    language: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class CodeResponse:
    """Response from Kimi K2 code operations"""
    content: str
    operation_type: str  # 'generate', 'refactor', 'debug', 'explain', 'diff'
    language: Optional[str] = None
    files_changed: List[str] = field(default_factory=list)
    diff: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    token_usage: Optional[Dict[str, int]] = None

class CodeContextExtractor:
    """Extracts code context from the current environment"""
    
    def __init__(self, config: KimiConfig):
        self.config = config
        self.project_root = Path.cwd()
    
    def extract_context(self, 
                       current_file: Optional[str] = None,
                       include_git: bool = True,
                       include_errors: bool = True) -> CodeContext:
        """Extract comprehensive code context"""
        context = CodeContext()
        
        try:
            # Get current file context
            if current_file and Path(current_file).exists():
                context.current_file = current_file
                context.current_file_content = self._read_file_safe(current_file)
                context.language = self._detect_language(current_file)
            
            # Get project files
            context.project_files = self._get_relevant_project_files()
            
            # Get git status if available
            if include_git:
                context.git_status = self._get_git_status()
            
            # Extract dependencies
            context.dependencies = self._extract_dependencies()
            
            logger.info(f"Extracted context: {len(context.project_files)} files, language: {context.language}")
            
        except Exception as e:
            logger.error(f"Error extracting code context: {e}")
        
        return context
    
    def _read_file_safe(self, file_path: str) -> Optional[str]:
        """Safely read file content"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > self.config.max_context_size:
                    # Truncate large files
                    content = content[:self.config.max_context_size] + "\n... (truncated)"
                return content
        except Exception as e:
            logger.warning(f"Could not read file {file_path}: {e}")
            return None
    
    def _get_relevant_project_files(self) -> List[Dict[str, Any]]:
        """Get relevant project files based on patterns"""
        files = []
        
        try:
            # Find files matching include patterns
            for pattern in self.config.include_patterns:
                for file_path in self.project_root.rglob(pattern):
                    if self._should_include_file(file_path):
                        file_info = {
                            'path': str(file_path.relative_to(self.project_root)),
                            'name': file_path.name,
                            'size': file_path.stat().st_size,
                            'language': self._detect_language(str(file_path)),
                            'modified': file_path.stat().st_mtime
                        }
                        
                        # Add content for small, important files
                        if file_path.stat().st_size < 10000:  # Less than 10KB
                            file_info['content'] = self._read_file_safe(str(file_path))
                        
                        files.append(file_info)
                        
                        # Limit number of files
                        if len(files) >= self.config.max_context_files:
                            break
            
            # Sort by importance (smaller, recently modified files first)
            files.sort(key=lambda f: (f['size'], -f['modified']))
            
        except Exception as e:
            logger.error(f"Error getting project files: {e}")
        
        return files[:self.config.max_context_files]
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Check if file should be included in context"""
        file_str = str(file_path)
        
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if pattern.replace('*', '') in file_str:
                return False
        
        # Skip binary files
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                if b'\x00' in chunk:  # Likely binary
                    return False
        except:
            return False
        
        return True
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.html': 'html',
            '.css': 'css',
            '.sql': 'sql',
            '.sh': 'bash',
            '.bash': 'bash',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.md': 'markdown'
        }
        
        ext = Path(file_path).suffix.lower()
        return extension_map.get(ext)
    
    def _get_git_status(self) -> Optional[str]:
        """Get git status if available"""
        try:
            import subprocess
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not get git status: {e}")
        return None
    
    def _extract_dependencies(self) -> List[str]:
        """Extract project dependencies from common files"""
        dependencies = []
        
        dep_files = [
            'requirements.txt', 'package.json', 'pom.xml', 'Cargo.toml',
            'go.mod', 'composer.json', 'Pipfile', 'pyproject.toml'
        ]
        
        for dep_file in dep_files:
            file_path = self.project_root / dep_file
            if file_path.exists():
                content = self._read_file_safe(str(file_path))
                if content:
                    dependencies.append(f"{dep_file}:\n{content[:1000]}")  # First 1000 chars
        
        return dependencies

class DiffGenerator:
    """Generates and formats code diffs for display"""
    
    @staticmethod
    def generate_unified_diff(original: str, modified: str, filename: str = "file") -> str:
        """Generate unified diff format"""
        try:
            import difflib
            original_lines = original.splitlines(keepends=True)
            modified_lines = modified.splitlines(keepends=True)
            
            diff = difflib.unified_diff(
                original_lines,
                modified_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
                lineterm=''
            )
            
            return ''.join(diff)
        except Exception as e:
            logger.error(f"Error generating diff: {e}")
            return f"Error generating diff: {e}"
    
    @staticmethod
    def format_diff_for_cursor(diff: str) -> str:
        """Format diff for better display in Cursor IDE"""
        lines = diff.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.startswith('+++') or line.startswith('---'):
                formatted_lines.append(f"📁 {line}")
            elif line.startswith('@@'):
                formatted_lines.append(f"📍 {line}")
            elif line.startswith('+'):
                formatted_lines.append(f"✅ {line}")
            elif line.startswith('-'):
                formatted_lines.append(f"❌ {line}")
            else:
                formatted_lines.append(f"   {line}")
        
        return '\n'.join(formatted_lines)

class KimiK2Agent:
    """Kimi K2 Code Agent for intelligent code operations"""
    
    def __init__(self, config: Optional[KimiConfig] = None):
        self.config = config or self._get_default_config()
        self.context_extractor = CodeContextExtractor(self.config)
        self.diff_generator = DiffGenerator()
        self.client: Optional[httpx.AsyncClient] = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'operations': {
                'generate': 0,
                'refactor': 0,
                'debug': 0,
                'explain': 0,
                'diff': 0
            }
        }
    
    def _get_default_config(self) -> KimiConfig:
        """Get default configuration with environment overrides"""
        config = KimiConfig()
        
        # Override with environment variables
        if api_key := os.getenv('KIMI_API_KEY'):
            config.api_key = api_key
        elif api_key := os.getenv('MOONSHOT_API_KEY'):
            config.api_key = api_key
        
        if base_url := os.getenv('KIMI_BASE_URL'):
            config.base_url = base_url
        
        if model_id := os.getenv('KIMI_MODEL_ID'):
            config.model_id = model_id
        
        return config
    
    async def initialize(self) -> None:
        """Initialize the agent"""
        if not self.config.api_key:
            raise ValueError("Kimi API key is required. Set KIMI_API_KEY or MOONSHOT_API_KEY environment variable.")
        
        # Initialize HTTP client
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("Kimi K2 Agent initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Kimi K2 Agent cleaned up")
    
    @asynccontextmanager
    async def _ensure_client(self):
        """Ensure client is initialized"""
        if not self.client:
            await self.initialize()
        try:
            yield self.client
        finally:
            pass  # Keep client alive for reuse
    
    def detect_code_request(self, message: str) -> bool:
        """Detect if message is a code-related request"""
        # Check for explicit #code tag
        if message.strip().startswith('#code'):
            return True
        
        # Check for code-related keywords
        code_keywords = [
            'code', 'function', 'class', 'method', 'variable', 'bug', 'error',
            'debug', 'refactor', 'implement', 'generate', 'create', 'write',
            'fix', 'optimize', 'review', 'test', 'algorithm', 'syntax',
            'import', 'export', 'module', 'package', 'library', 'framework',
            'api', 'endpoint', 'database', 'query', 'script', 'automation'
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in code_keywords)
    
    def extract_code_request(self, message: str) -> Tuple[str, str]:
        """Extract operation type and clean request from message"""
        # Remove #code tag if present
        clean_message = re.sub(r'^#code\s*', '', message.strip())
        
        # Detect operation type
        operation_patterns = {
            'generate': r'\b(generate|create|write|build|make)\b',
            'refactor': r'\b(refactor|improve|optimize|restructure|clean up)\b',
            'debug': r'\b(debug|fix|solve|resolve|troubleshoot|error|bug)\b',
            'explain': r'\b(explain|describe|what does|how does|understand)\b',
            'diff': r'\b(diff|compare|difference|changes|modify)\b'
        }
        
        message_lower = clean_message.lower()
        for operation, pattern in operation_patterns.items():
            if re.search(pattern, message_lower):
                return operation, clean_message
        
        # Default to generate if no specific operation detected
        return 'generate', clean_message
    
    async def process_code_request(self, 
                                 message: str,
                                 current_file: Optional[str] = None,
                                 error_context: Optional[str] = None) -> CodeResponse:
        """Process a code-related request"""
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # Extract operation and clean message
            operation, clean_request = self.extract_code_request(message)
            self.stats['operations'][operation] += 1
            
            # Extract code context
            context = self.context_extractor.extract_context(
                current_file=current_file,
                include_git=True,
                include_errors=bool(error_context)
            )
            
            if error_context:
                context.error_content = error_context
            
            # Generate response
            response = await self._generate_code_response(operation, clean_request, context)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            response.execution_time = execution_time
            
            # Update statistics
            self.stats['successful_requests'] += 1
            current_avg = self.stats['avg_response_time']
            successful = self.stats['successful_requests']
            self.stats['avg_response_time'] = ((current_avg * (successful - 1)) + execution_time) / successful
            
            logger.info(f"Code request processed: {operation} ({execution_time:.2f}s)")
            return response
            
        except Exception as e:
            self.stats['failed_requests'] += 1
            logger.error(f"Error processing code request: {e}")
            
            return CodeResponse(
                content=f"I apologize, but I encountered an error processing your code request: {str(e)}",
                operation_type=operation if 'operation' in locals() else 'unknown',
                execution_time=time.time() - start_time
            )
    
    async def _generate_code_response(self, 
                                    operation: str,
                                    request: str,
                                    context: CodeContext) -> CodeResponse:
        """Generate response using Kimi K2 API"""
        async with self._ensure_client() as client:
            # Build prompt based on operation type
            system_prompt = self._build_system_prompt(operation, context)
            user_prompt = self._build_user_prompt(request, context)
            
            # Prepare API request
            payload = {
                "model": self.config.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature
            }
            
            # Make API call
            response = await client.post(
                f"{self.config.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extract token usage if available
            token_usage = result.get('usage', {})
            if token_usage:
                self.stats['total_tokens'] += token_usage.get('total_tokens', 0)
            
            # Generate diff if applicable
            diff = None
            if operation in ['refactor', 'debug', 'diff'] and context.current_file_content:
                # Try to extract new code from response
                new_code = self._extract_code_from_response(content)
                if new_code:
                    diff = self.diff_generator.generate_unified_diff(
                        context.current_file_content, 
                        new_code, 
                        context.current_file or "file"
                    )
            
            return CodeResponse(
                content=content,
                operation_type=operation,
                language=context.language,
                files_changed=[context.current_file] if context.current_file else [],
                diff=diff,
                token_usage=token_usage
            )
    
    def _build_system_prompt(self, operation: str, context: CodeContext) -> str:
        """Build system prompt based on operation and context"""
        base_prompt = "You are Kimi K2, an expert code assistant. "
        
        operation_prompts = {
            'generate': "Generate clean, well-documented, and efficient code. Follow best practices and include comprehensive comments explaining the logic.",
            'refactor': "Improve the provided code by enhancing readability, performance, and maintainability. Preserve functionality while applying best practices.",
            'debug': "Analyze the code and error context to identify issues. Provide clear explanations of problems and working solutions.",
            'explain': "Provide detailed explanations of how the code works, including its purpose, logic flow, and any complex concepts.",
            'diff': "Compare code versions and highlight key differences, improvements, and potential impacts of changes."
        }
        
        prompt = base_prompt + operation_prompts.get(operation, operation_prompts['generate'])
        
        if context.language:
            prompt += f" The primary language is {context.language}."
        
        if context.project_files:
            prompt += f" This is part of a larger project with {len(context.project_files)} files."
        
        prompt += " Always provide production-ready code with proper error handling."
        
        return prompt
    
    def _build_user_prompt(self, request: str, context: CodeContext) -> str:
        """Build user prompt with context"""
        prompt_parts = [f"Request: {request}"]
        
        # Add current file context
        if context.current_file and context.current_file_content:
            prompt_parts.append(f"\nCurrent file ({context.current_file}):")
            prompt_parts.append(f"```{context.language or ''}")
            prompt_parts.append(context.current_file_content)
            prompt_parts.append("```")
        
        # Add error context if available
        if context.error_content:
            prompt_parts.append(f"\nError context:")
            prompt_parts.append(context.error_content)
        
        # Add relevant project files
        if context.project_files:
            prompt_parts.append(f"\nProject structure ({len(context.project_files)} files):")
            for file_info in context.project_files[:5]:  # Limit to 5 files
                prompt_parts.append(f"- {file_info['path']} ({file_info.get('language', 'unknown')})")
                if file_info.get('content') and len(file_info['content']) < 1000:
                    prompt_parts.append(f"  Content preview: {file_info['content'][:500]}...")
        
        # Add git status if available
        if context.git_status:
            prompt_parts.append(f"\nGit status:")
            prompt_parts.append(context.git_status)
        
        # Add dependencies
        if context.dependencies:
            prompt_parts.append(f"\nProject dependencies:")
            for dep in context.dependencies[:3]:  # Limit to 3 dependency files
                prompt_parts.append(dep)
        
        return '\n'.join(prompt_parts)
    
    def _extract_code_from_response(self, content: str) -> Optional[str]:
        """Extract code blocks from response"""
        # Look for code blocks in markdown format
        # Handle various code block formats: ```python, ```, etc.
        code_block_pattern = r'```(?:\w+)?\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, content, re.DOTALL)
        
        if matches:
            # Return the largest code block, stripped of extra whitespace
            largest = max(matches, key=len)
            return largest.strip()
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        return {
            'performance': self.stats.copy(),
            'configuration': {
                'model_id': self.config.model_id,
                'base_url': self.config.base_url,
                'max_tokens': self.config.max_tokens,
                'temperature': self.config.temperature,
                'supported_languages': self.config.supported_languages,
                'max_context_files': self.config.max_context_files
            },
            'context_info': {
                'project_root': str(self.context_extractor.project_root),
                'include_patterns': self.config.include_patterns,
                'exclude_patterns': self.config.exclude_patterns
            }
        }
    
    def reset_stats(self) -> None:
        """Reset agent statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'operations': {
                'generate': 0,
                'refactor': 0,
                'debug': 0,
                'explain': 0,
                'diff': 0
            }
        }
        logger.info("Agent statistics reset")

# Factory function for easy integration
def create_kimi_agent(config: Optional[KimiConfig] = None) -> KimiK2Agent:
    """Create a KimiK2Agent instance with optional configuration"""
    return KimiK2Agent(config) 