#!/usr/bin/env python3
"""
Integration Test Dependencies Checker
Validates all required dependencies for integration testing
"""

import sys
import importlib
from typing import List, Tuple

def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies are available"""
    
    required_packages = [
        ('pytest', 'pytest'),
        ('asyncio', 'asyncio'),
        ('unittest.mock', 'unittest.mock'),
        ('numpy', 'numpy'),
        ('psutil', 'psutil'),
        ('pathlib', 'pathlib'),
        ('difflib', 'difflib'),
        ('base64', 'base64'),
        ('io', 'io'),
        ('time', 'time'),
        ('json', 'json'),
        ('argparse', 'argparse')
    ]
    
    optional_packages = [
        ('PIL', 'Pillow'),
        ('pytesseract', 'pytesseract'),
        ('nltk', 'nltk')
    ]
    
    missing_required = []
    missing_optional = []
    
    print("🔍 Checking Integration Test Dependencies...")
    print("=" * 50)
    
    # Check required packages
    for module_name, package_name in required_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_required.append(package_name)
            print(f"❌ {package_name} (REQUIRED)")
    
    # Check optional packages
    for module_name, package_name in optional_packages:
        try:
            importlib.import_module(module_name)
            print(f"✅ {package_name} (optional)")
        except ImportError:
            missing_optional.append(package_name)
            print(f"⚠️  {package_name} (optional)")
    
    print("\n" + "=" * 50)
    
    if missing_required:
        print("❌ Missing required dependencies:")
        for package in missing_required:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_required))
        return False, missing_required
    
    if missing_optional:
        print("⚠️  Missing optional dependencies (some tests may be limited):")
        for package in missing_optional:
            print(f"   - {package}")
        print("\nInstall with: pip install " + " ".join(missing_optional))
    
    print("✅ All required dependencies available")
    print("🚀 Ready for integration testing")
    
    return True, []

if __name__ == '__main__':
    success, missing = check_dependencies()
    sys.exit(0 if success else 1) 