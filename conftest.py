#!/usr/bin/env python3
"""
Root Test Configuration for Sovereign 4.0 Voice Assistant
Provides global pytest configuration and plugins
"""

# Global pytest plugins configuration
pytest_plugins = ('pytest_asyncio',)

import os
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'DEBUG' 