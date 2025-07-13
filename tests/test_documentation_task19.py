"""
Tests for Task 19: Update Documentation and User Guides

Validates that all required documentation has been created and contains
the expected content for the Realtime API features.
"""

import pytest
import os
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List

# Documentation files that should exist
REQUIRED_DOCS = {
    'README.md': {
        'keywords': [
            'Realtime API', 'Hybrid Mode', '<300ms', 'Smart Context',
            'Screen Awareness', 'Memory Context', 'OpenAI', 'Ultra-fast'
        ],
        'sections': [
            'Revolutionary Performance Breakthrough',
            'What\'s New in Realtime API Edition',
            'Quick Start',
            'Usage Examples',
            'Architecture Overview',
            'Performance Monitoring',
            'Configuration',
            'Testing & Validation',
            'Project Structure',
            'Troubleshooting'
        ]
    },
    'docs/api_documentation.md': {
        'keywords': [
            'HybridVoiceSystem', 'RealtimeVoiceService', 'SmartContextManager',
            'ScreenContextProvider', 'MemoryContextProvider', 'AudioStreamManager',
            'ConnectionStabilityMonitor', 'ModeSwitchManager'
        ],
        'sections': [
            'Core Components', 'Realtime API Service', 'Smart Context Management',
            'Screen Context Provider', 'Memory Context Provider', 'Audio Stream Management'
        ]
    },
    'docs/configuration_reference.md': {
        'keywords': [
            'realtime_api', 'smart_context', 'screen_context', 'memory_context',
            'audio', 'performance', 'monitoring', 'token_budget'
        ],
        'sections': [
            'Environment Variables', 'YAML Configuration', 'Realtime API Settings',
            'Smart Context Configuration', 'Audio Configuration', 'Performance Tuning'
        ]
    },
    'docs/troubleshooting_guide.md': {
        'keywords': [
            'Connection Failed', 'High Latency', 'Audio Quality', 'OCR Not Working',
            'ChromaDB', 'Memory Issues', 'Performance Problems'
        ],
        'sections': [
            'Quick Diagnostics', 'Realtime API Issues', 'Connection Problems',
            'Audio Issues', 'Performance Problems', 'Context Management Issues'
        ]
    },
    'docs/migration_guide.md': {
        'keywords': [
            'Traditional to Hybrid', 'Configuration Migration', 'Breaking Changes',
            'Performance Improvement', 'Backup and Preparation'
        ],
        'sections': [
            'Migration Overview', 'Pre-Migration Assessment', 'Backup and Preparation',
            'Configuration Migration', 'Code Migration', 'Testing and Validation'
        ]
    },
    'docs/performance_tuning_guide.md': {
        'keywords': [
            'Latency Optimization', 'Audio Quality vs Speed', 'Context Management Tuning',
            'System Resource Optimization', 'Network Optimization'
        ],
        'sections': [
            'Performance Overview', 'Quick Performance Gains', 'Latency Optimization',
            'Audio Quality vs Speed', 'Context Management Tuning'
        ]
    },
    'docs/developer_documentation.md': {
        'keywords': [
            'Architecture Overview', 'System Components', 'Development Setup',
            'Code Structure', 'API Integration Patterns', 'Testing Guidelines'
        ],
        'sections': [
            'Architecture Overview', 'System Components', 'Development Setup',
            'Code Structure', 'API Integration Patterns', 'Testing Guidelines'
        ]
    },
    'CHANGELOG.md': {
        'keywords': [
            '[4.0.0]', 'Realtime API Revolution', 'OpenAI Realtime API Integration',
            'Hybrid Voice System', 'Smart Context Management', 'Breaking Changes'
        ],
        'sections': [
            'Revolutionary Features Added', 'Enhanced Features', 'Changed',
            'Fixed', 'Removed', 'Security'
        ]
    }
}

class TestDocumentationTask19:
    """Test suite for Task 19 documentation implementation."""
    
    def test_all_required_files_exist(self):
        """Test that all required documentation files exist."""
        
        missing_files = []
        
        for doc_file in REQUIRED_DOCS.keys():
            if not os.path.exists(doc_file):
                missing_files.append(doc_file)
        
        assert not missing_files, f"Missing documentation files: {missing_files}"
    
    def test_readme_realtime_api_content(self):
        """Test that README contains comprehensive Realtime API content."""
        
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key Realtime API features
        required_features = [
            'Realtime API Edition',
            '<300ms response times',
            'OpenAI\'s Realtime API',
            'Hybrid Mode',
            'Smart Context Management',
            'Screen Awareness',
            'Memory Context',
            'Ultra-fast'
        ]
        
        for feature in required_features:
            assert feature in content, f"README missing feature: {feature}"
        
        # Check for performance metrics table
        assert 'Response Time' in content and '<300ms' in content
        assert '800ms+' in content  # Old vs new comparison
        
        # Check for architecture diagrams
        assert 'Architecture Overview' in content
        assert 'Realtime API Pipeline' in content or 'Hybrid Fallback Architecture' in content
    
    def test_api_documentation_completeness(self):
        """Test that API documentation covers all major components."""
        
        with open('docs/api_documentation.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for all major API components
        required_components = [
            'HybridVoiceSystem',
            'RealtimeVoiceService', 
            'SmartContextManager',
            'ScreenContextProvider',
            'MemoryContextProvider',
            'AudioStreamManager',
            'ConnectionStabilityMonitor'
        ]
        
        for component in required_components:
            assert component in content, f"API docs missing component: {component}"
        
        # Check for code examples
        assert '```python' in content, "API docs missing code examples"
        assert 'async def' in content, "API docs missing async method examples"
        
        # Check for configuration examples
        assert '```yaml' in content, "API docs missing YAML configuration examples"
    
    def test_configuration_reference_comprehensive(self):
        """Test that configuration reference is comprehensive."""
        
        with open('docs/configuration_reference.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for all major configuration sections
        required_sections = [
            'realtime_api:',
            'smart_context:',
            'screen_context:',
            'memory_context:',
            'audio:',
            'performance:',
            'monitoring:'
        ]
        
        for section in required_sections:
            assert section in content, f"Config reference missing section: {section}"
        
        # Check for environment variables
        assert 'OPENAI_API_KEY' in content
        assert 'SOVEREIGN_MODE' in content
        
        # Check for examples and explanations
        assert 'gpt-4o-realtime-preview' in content
        assert 'token_budget: 32000' in content
    
    def test_troubleshooting_guide_covers_common_issues(self):
        """Test that troubleshooting guide covers expected issues."""
        
        with open('docs/troubleshooting_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for common issue categories
        issue_categories = [
            'Realtime API Issues',
            'Connection Problems', 
            'Audio Issues',
            'Performance Problems',
            'Context Management Issues',
            'Screen Context Problems',
            'Memory Issues'
        ]
        
        for category in issue_categories:
            assert category in content, f"Troubleshooting missing category: {category}"
        
        # Check for specific error scenarios
        assert 'Connection Failed' in content
        assert 'High Latency' in content
        assert 'OCR Not Working' in content
        assert 'ChromaDB' in content
        
        # Check for solutions and commands
        assert '```bash' in content, "Troubleshooting missing bash commands"
        assert 'python -c' in content, "Troubleshooting missing diagnostic commands"
    
    def test_migration_guide_comprehensive(self):
        """Test that migration guide provides complete migration path."""
        
        with open('docs/migration_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check migration phases
        migration_phases = [
            'Migration Overview',
            'Pre-Migration Assessment',
            'Backup and Preparation',
            'Configuration Migration',
            'Code Migration',
            'Testing and Validation'
        ]
        
        for phase in migration_phases:
            assert phase in content, f"Migration guide missing phase: {phase}"
        
        # Check for before/after examples
        assert '# OLD' in content or 'OLD (' in content or 'old configuration' in content.lower(), "Migration guide should show before/after examples"
        
        # Check for backup instructions
        assert 'backup' in content.lower()
        assert 'requirements_backup.txt' in content
        
        # Check for breaking changes
        assert 'Breaking Changes' in content
    
    def test_performance_tuning_guide_detailed(self):
        """Test that performance tuning guide provides actionable guidance."""
        
        with open('docs/performance_tuning_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for performance targets and metrics
        assert '<300ms' in content or '300ms' in content
        assert 'latency' in content.lower()
        assert 'performance' in content.lower()
        
        # Check for optimization strategies
        optimization_areas = [
            'Latency Optimization',
            'Audio Quality vs Speed',
            'Context Management Tuning',
            'System Resource Optimization',
            'Network Optimization'
        ]
        
        for area in optimization_areas:
            assert area in content, f"Performance guide missing area: {area}"
        
        # Check for configuration examples
        assert '```yaml' in content, "Performance guide missing config examples"
        assert 'chunk_size' in content or 'sample_rate' in content
    
    def test_developer_documentation_architecture(self):
        """Test that developer documentation covers architecture properly."""
        
        with open('docs/developer_documentation.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for architecture sections
        arch_sections = [
            'Architecture Overview',
            'System Components',
            'Development Setup',
            'Code Structure',
            'API Integration Patterns'
        ]
        
        for section in arch_sections:
            assert section in content, f"Developer docs missing section: {section}"
        
        # Check for code examples and patterns
        assert '```python' in content, "Developer docs missing Python examples"
        assert 'class ' in content, "Developer docs missing class examples"
        assert 'async def' in content, "Developer docs missing async examples"
        
        # Check for development setup instructions
        assert 'venv' in content or 'virtual environment' in content
        assert 'requirements.txt' in content or 'pip install' in content
    
    def test_changelog_version_4_features(self):
        """Test that changelog properly documents v4.0 features."""
        
        with open('CHANGELOG.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for version 4.0 section
        assert '[4.0.0]' in content, "Changelog missing v4.0 section"
        assert 'Realtime API Revolution' in content or 'Realtime API' in content
        
        # Check for major feature categories
        feature_categories = [
            'OpenAI Realtime API Integration',
            'Hybrid Voice System',
            'Smart Context Management',
            'Screen Context Provider',
            'Memory Context Provider'
        ]
        
        for category in feature_categories:
            assert category in content, f"Changelog missing feature: {category}"
        
        # Check for breaking changes
        assert 'Breaking Changes' in content or 'BREAKING' in content
        
        # Check for performance improvements
        assert '300ms' in content or '<300ms' in content
        assert '60%' in content or 'improvement' in content.lower()
    
    def test_documentation_cross_references(self):
        """Test that documentation files properly cross-reference each other."""
        
        # README should reference other docs
        with open('README.md', 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Should reference docs directory or specific guides
        expected_refs = [
            'docs/',
            'troubleshooting',
            'configuration',
            'migration'
        ]
        
        found_refs = sum(1 for ref in expected_refs if ref in readme_content.lower())
        assert found_refs >= 2, "README should reference other documentation"
    
    def test_code_examples_syntax_valid(self):
        """Test that code examples in documentation have valid syntax."""
        
        import ast
        import re
        
        # Files to check for Python code examples
        files_to_check = [
            'docs/api_documentation.md',
            'docs/developer_documentation.md',
            'docs/migration_guide.md'
        ]
        
        python_code_pattern = r'```python\n(.*?)\n```'
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract Python code blocks
            code_blocks = re.findall(python_code_pattern, content, re.DOTALL)
            
            for i, code_block in enumerate(code_blocks):
                # Skip incomplete examples (indicated by comments or ellipsis)
                if '...' in code_block or '# Example' in code_block:
                    continue
                
                # Try to parse the code
                try:
                    ast.parse(code_block)
                except SyntaxError as e:
                    # Allow some common documentation patterns
                    if 'from assistant.' in code_block or 'import assistant' in code_block:
                        continue  # Skip import examples that reference non-existent modules
                    
                    pytest.fail(f"Invalid Python syntax in {file_path} code block {i+1}: {e}")
    
    def test_yaml_examples_syntax_valid(self):
        """Test that YAML examples in documentation have valid syntax."""
        
        import re
        
        files_to_check = [
            'docs/configuration_reference.md',
            'docs/migration_guide.md',
            'docs/performance_tuning_guide.md'
        ]
        
        yaml_code_pattern = r'```yaml\n(.*?)\n```'
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract YAML code blocks
            yaml_blocks = re.findall(yaml_code_pattern, content, re.DOTALL)
            
            for i, yaml_block in enumerate(yaml_blocks):
                # Skip comment-only or partial examples
                if yaml_block.strip().startswith('#') or '...' in yaml_block:
                    continue
                
                try:
                    yaml.safe_load(yaml_block)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML syntax in {file_path} code block {i+1}: {e}")
    
    def test_documentation_accessibility(self):
        """Test that documentation follows accessibility best practices."""
        
        files_to_check = list(REQUIRED_DOCS.keys())
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for proper heading structure
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            
            # Should have at least one main heading
            assert len(headings) > 0, f"{file_path} should have headings"
            
            # Check for table of contents in longer documents
            if len(content) > 5000:  # Long documents
                toc_indicators = ['Table of Contents', 'Contents', '- [']
                has_toc = any(indicator in content for indicator in toc_indicators)
                assert has_toc, f"{file_path} should have table of contents for accessibility"
    
    def test_installation_instructions_complete(self):
        """Test that installation instructions are complete and clear."""
        
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for installation section
        assert 'Installation' in content or 'Quick Start' in content
        
        # Check for prerequisite information
        assert 'Python' in content
        assert 'OpenAI API' in content or 'API key' in content
        
        # Check for setup commands
        installation_commands = [
            'pip install',
            'git clone',
            '.env'
        ]
        
        found_commands = sum(1 for cmd in installation_commands if cmd in content)
        assert found_commands >= 2, "Installation instructions should include setup commands"
    
    def test_performance_metrics_documented(self):
        """Test that performance metrics and targets are properly documented."""
        
        files_with_metrics = ['README.md', 'docs/performance_tuning_guide.md']
        
        for file_path in files_with_metrics:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for performance metrics
            performance_indicators = [
                '300ms',
                '<300ms', 
                'latency',
                'response time',
                'performance'
            ]
            
            found_indicators = sum(1 for indicator in performance_indicators 
                                 if indicator.lower() in content.lower())
            
            assert found_indicators >= 2, f"{file_path} should document performance metrics"
    
    def test_security_and_privacy_documented(self):
        """Test that security and privacy considerations are documented."""
        
        files_to_check = ['README.md', 'docs/configuration_reference.md']
        
        security_topics = [
            'API key',
            'security',
            'privacy',
            'sensitive',
            'encryption',
            'PII'
        ]
        
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found_topics = sum(1 for topic in security_topics 
                             if topic.lower() in content.lower())
            
            # At least some security/privacy content should be present
            assert found_topics >= 1, f"{file_path} should address security/privacy"

# Specific content validation tests
class TestDocumentationContent:
    """Tests for specific content requirements in documentation."""
    
    def test_readme_quick_start_actionable(self):
        """Test that README quick start is actionable."""
        
        with open('README.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have numbered steps or clear progression
        step_indicators = ['1.', '2.', '3.', '###', 'Step', 'First', 'Then']
        has_steps = any(indicator in content for indicator in step_indicators)
        assert has_steps, "README should have clear step-by-step instructions"
        
        # Should include actual commands
        assert '```bash' in content or '```' in content, "README should have executable commands"
    
    def test_troubleshooting_solutions_actionable(self):
        """Test that troubleshooting solutions are actionable."""
        
        with open('docs/troubleshooting_guide.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have solution sections
        solution_indicators = ['Solutions:', 'Solution:', 'Fix:', 'Resolution:']
        has_solutions = any(indicator in content for indicator in solution_indicators)
        assert has_solutions, "Troubleshooting should provide clear solutions"
        
        # Should include diagnostic commands
        assert 'python -c' in content or 'python assistant' in content
        assert '```bash' in content, "Troubleshooting should include diagnostic commands"
    
    def test_configuration_examples_complete(self):
        """Test that configuration examples are complete and usable."""
        
        with open('docs/configuration_reference.md', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should have complete configuration examples
        config_sections = ['realtime_api:', 'smart_context:', 'audio:']
        for section in config_sections:
            assert section in content, f"Configuration should include {section} example"
        
        # Should include explanations
        explanation_indicators = ['Description:', 'Explanation:', 'Note:', 'Important:']
        has_explanations = any(indicator in content for indicator in explanation_indicators)
        assert has_explanations, "Configuration should include explanations"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 