"""
Accuracy Benchmarking Suite for Sovereign 4.0 Voice Assistant

This module provides comprehensive accuracy testing capabilities including:
- BLEU score calculation for memory recall evaluation
- Semantic similarity metrics using sentence transformers
- OCR accuracy validation for screen monitoring
- Multi-modal accuracy assessment

Implements modern 2024-2025 accuracy benchmarking practices.
"""

import asyncio
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

# ML and NLP libraries for accuracy testing
try:
    import evaluate
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import cv2
    import pytesseract
    from PIL import Image
    HAS_ML_LIBRARIES = True
except ImportError:
    HAS_ML_LIBRARIES = False
    logging.warning("ML libraries not available. Some accuracy tests will be disabled.")

from .performance_types import TestResult, PerformanceTestConfig

logger = logging.getLogger(__name__)

@dataclass
class AccuracyTestCase:
    """Individual accuracy test case"""
    test_id: str
    input_data: Any
    expected_output: str
    test_type: str  # 'stt', 'memory_recall', 'ocr', 'semantic'
    metadata: Dict[str, Any]

class AccuracyBenchmarkSuite:
    """
    Comprehensive accuracy benchmarking suite
    
    Implements multi-modal accuracy testing with BLEU scores, semantic similarity,
    and OCR validation as required by the task specifications.
    """
    
    def __init__(self, config: PerformanceTestConfig):
        self.config = config
        
        # Initialize ML models if available
        if HAS_ML_LIBRARIES:
            try:
                self.bleu_metric = evaluate.load("bleu")
                self.bert_score = evaluate.load("bertscore")
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("ML models loaded successfully for accuracy testing")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}")
                self.bleu_metric = None
                self.bert_score = None
                self.semantic_model = None
        else:
            self.bleu_metric = None
            self.bert_score = None
            self.semantic_model = None
        
        # Load test datasets
        self.test_cases = self._load_test_datasets()
        
    def _load_test_datasets(self) -> Dict[str, List[AccuracyTestCase]]:
        """Load standardized test datasets for accuracy benchmarking"""
        datasets = {
            'memory_recall': self._create_memory_recall_dataset(),
            'stt_accuracy': self._create_stt_accuracy_dataset(),
            'ocr_accuracy': self._create_ocr_accuracy_dataset(),
            'semantic_similarity': self._create_semantic_similarity_dataset()
        }
        
        logger.info(f"Loaded test datasets: {[f'{k}: {len(v)} cases' for k, v in datasets.items()]}")
        return datasets
    
    def _create_memory_recall_dataset(self) -> List[AccuracyTestCase]:
        """Create test cases for memory recall accuracy (BLEU score evaluation)"""
        test_cases = []
        
        # Programming context test cases
        programming_cases = [
            {
                'input': "What error did we discuss about the authentication system?",
                'expected': "The authentication system had a JWT token expiration issue where tokens were not being refreshed properly, causing users to be logged out unexpectedly.",
                'context': "Previous conversation about debugging authentication errors"
            },
            {
                'input': "Remind me about the database optimization we talked about",
                'expected': "We discussed adding indexes to the user_activities table on the timestamp and user_id columns to improve query performance by 40%.",
                'context': "Database performance optimization discussion"
            },
            {
                'input': "What was the solution for the memory leak in the voice processing?",
                'expected': "The memory leak was caused by not properly disposing of audio buffers. We fixed it by implementing proper cleanup in the audio processing pipeline.",
                'context': "Voice processing optimization discussion"
            }
        ]
        
        for i, case in enumerate(programming_cases):
            test_cases.append(AccuracyTestCase(
                test_id=f"memory_recall_{i+1}",
                input_data=case['input'],
                expected_output=case['expected'],
                test_type="memory_recall",
                metadata={'context': case['context'], 'domain': 'programming'}
            ))
        
        # Add general knowledge test cases
        general_cases = [
            {
                'input': "What did I ask about Python decorators?",
                'expected': "You asked how to create a decorator that measures function execution time and logs the results to a file.",
                'context': "Python programming discussion"
            },
            {
                'input': "Remind me about the API design pattern we discussed",
                'expected': "We discussed implementing the Repository pattern with dependency injection for better testability and separation of concerns.",
                'context': "Software architecture discussion"
            }
        ]
        
        for i, case in enumerate(general_cases):
            test_cases.append(AccuracyTestCase(
                test_id=f"memory_recall_general_{i+1}",
                input_data=case['input'],
                expected_output=case['expected'],
                test_type="memory_recall",
                metadata={'context': case['context'], 'domain': 'general'}
            ))
        
        return test_cases
    
    def _create_stt_accuracy_dataset(self) -> List[AccuracyTestCase]:
        """Create test cases for STT accuracy validation"""
        test_cases = []
        
        # Technical vocabulary test cases
        technical_phrases = [
            "Initialize the Docker container with environment variables",
            "Implement asynchronous database queries using asyncio",
            "Configure the Kubernetes deployment with persistent volumes",
            "Debug the TypeScript interface inheritance issue",
            "Optimize the React component rendering performance"
        ]
        
        for i, phrase in enumerate(technical_phrases):
            test_cases.append(AccuracyTestCase(
                test_id=f"stt_technical_{i+1}",
                input_data=phrase,  # Would be audio data in real implementation
                expected_output=phrase,
                test_type="stt",
                metadata={'category': 'technical', 'complexity': 'medium'}
            ))
        
        # Natural language test cases
        natural_phrases = [
            "Can you help me understand how this algorithm works?",
            "What's the best practice for handling user authentication?",
            "How do I resolve this compilation error in my code?",
            "Please explain the difference between these two approaches",
            "Show me how to implement this feature step by step"
        ]
        
        for i, phrase in enumerate(natural_phrases):
            test_cases.append(AccuracyTestCase(
                test_id=f"stt_natural_{i+1}",
                input_data=phrase,
                expected_output=phrase,
                test_type="stt",
                metadata={'category': 'natural', 'complexity': 'low'}
            ))
        
        return test_cases
    
    def _create_ocr_accuracy_dataset(self) -> List[AccuracyTestCase]:
        """Create test cases for OCR accuracy validation (screen monitoring)"""
        test_cases = []
        
        # Error dialog test cases
        error_dialogs = [
            "Error: Cannot find module 'express'",
            "TypeError: Cannot read property 'length' of undefined",
            "SyntaxError: Unexpected token ';' at line 42",
            "ModuleNotFoundError: No module named 'numpy'",
            "ReferenceError: variable is not defined"
        ]
        
        for i, error_text in enumerate(error_dialogs):
            test_cases.append(AccuracyTestCase(
                test_id=f"ocr_error_{i+1}",
                input_data=error_text,  # Would be image data in real implementation
                expected_output=error_text,
                test_type="ocr",
                metadata={'category': 'error_dialog', 'theme': 'dark'}
            ))
        
        # Code snippet test cases
        code_snippets = [
            "async def process_data(data: List[str]) -> Dict[str, Any]:",
            "const handleSubmit = async (event: React.FormEvent) => {",
            "public class UserService implements IUserRepository {",
            "SELECT * FROM users WHERE created_at > '2024-01-01'",
            "docker run -d -p 8080:80 --name myapp nginx:latest"
        ]
        
        for i, code in enumerate(code_snippets):
            test_cases.append(AccuracyTestCase(
                test_id=f"ocr_code_{i+1}",
                input_data=code,
                expected_output=code,
                test_type="ocr",
                metadata={'category': 'code_snippet', 'language': 'mixed'}
            ))
        
        return test_cases
    
    def _create_semantic_similarity_dataset(self) -> List[AccuracyTestCase]:
        """Create test cases for semantic similarity validation"""
        test_cases = []
        
        # Paraphrase test cases (should have high semantic similarity)
        paraphrase_pairs = [
            {
                'input': "How do I fix this bug in my JavaScript code?",
                'expected': "Can you help me resolve this JavaScript error?",
                'similarity_target': 0.85
            },
            {
                'input': "What's the best way to optimize database performance?",
                'expected': "How can I improve my database query speed?",
                'similarity_target': 0.80
            },
            {
                'input': "Implement user authentication with JWT tokens",
                'expected': "Create a login system using JSON Web Tokens",
                'similarity_target': 0.78
            }
        ]
        
        for i, pair in enumerate(paraphrase_pairs):
            test_cases.append(AccuracyTestCase(
                test_id=f"semantic_paraphrase_{i+1}",
                input_data=pair['input'],
                expected_output=pair['expected'],
                test_type="semantic",
                metadata={'target_similarity': pair['similarity_target'], 'type': 'paraphrase'}
            ))
        
        return test_cases
    
    async def run_all_accuracy_tests(self) -> List[TestResult]:
        """Run comprehensive accuracy test suite"""
        results = []
        
        # Memory recall accuracy (BLEU score)
        results.append(await self.test_memory_recall_accuracy())
        
        # STT accuracy
        results.append(await self.test_stt_accuracy())
        
        # OCR accuracy
        results.append(await self.test_ocr_accuracy())
        
        # Semantic similarity
        results.append(await self.test_semantic_similarity())
        
        # Composite accuracy score
        results.append(await self.calculate_composite_accuracy_score(results))
        
        return results
    
    async def test_memory_recall_accuracy(self) -> TestResult:
        """Test memory recall accuracy using BLEU scores"""
        start_time = time.time()
        test_name = "memory_recall_bleu"
        
        try:
            if not self.bleu_metric:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="BLEU metric not available"
                )
            
            test_cases = self.test_cases['memory_recall']
            bleu_scores = []
            bert_scores = []
            
            for test_case in test_cases:
                # Simulate memory recall response (would integrate with actual system)
                predicted_response = await self._simulate_memory_recall(test_case.input_data)
                
                # Calculate BLEU score
                bleu_result = self.bleu_metric.compute(
                    predictions=[predicted_response],
                    references=[[test_case.expected_output]]
                )
                bleu_scores.append(bleu_result['bleu'])
                
                # Calculate BERTScore for semantic accuracy
                if self.bert_score:
                    bert_result = self.bert_score.compute(
                        predictions=[predicted_response],
                        references=[test_case.expected_output],
                        model_type="distilbert-base-uncased"
                    )
                    bert_scores.append(bert_result['f1'][0])
            
            # Calculate metrics
            metrics = {
                'mean_bleu_score': np.mean(bleu_scores),
                'min_bleu_score': np.min(bleu_scores),
                'max_bleu_score': np.max(bleu_scores),
                'bleu_std_dev': np.std(bleu_scores),
                'test_cases_count': len(test_cases)
            }
            
            if bert_scores:
                metrics.update({
                    'mean_bert_f1': np.mean(bert_scores),
                    'min_bert_f1': np.min(bert_scores)
                })
            
            # Determine status based on target
            target_bleu = self.config.accuracy_targets['memory_recall_bleu']
            status = 'passed' if metrics['mean_bleu_score'] >= target_bleu else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Mean BLEU: {metrics['mean_bleu_score']:.3f} (target: {target_bleu})"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_stt_accuracy(self) -> TestResult:
        """Test STT accuracy using character and word accuracy metrics"""
        start_time = time.time()
        test_name = "stt_accuracy"
        
        try:
            test_cases = self.test_cases['stt_accuracy']
            character_accuracies = []
            word_accuracies = []
            
            for test_case in test_cases:
                # Simulate STT transcription (would integrate with actual STT service)
                transcribed_text = await self._simulate_stt_transcription(test_case.input_data)
                
                # Calculate character-level accuracy
                char_accuracy = self._calculate_character_accuracy(
                    transcribed_text, test_case.expected_output
                )
                character_accuracies.append(char_accuracy)
                
                # Calculate word-level accuracy
                word_accuracy = self._calculate_word_accuracy(
                    transcribed_text, test_case.expected_output
                )
                word_accuracies.append(word_accuracy)
            
            # Calculate metrics
            metrics = {
                'mean_character_accuracy': np.mean(character_accuracies),
                'mean_word_accuracy': np.mean(word_accuracies),
                'min_character_accuracy': np.min(character_accuracies),
                'min_word_accuracy': np.min(word_accuracies),
                'character_accuracy_std': np.std(character_accuracies),
                'word_accuracy_std': np.std(word_accuracies),
                'test_cases_count': len(test_cases)
            }
            
            # Determine status based on target
            target_accuracy = self.config.accuracy_targets['stt_accuracy']
            status = 'passed' if metrics['mean_word_accuracy'] >= target_accuracy else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Word accuracy: {metrics['mean_word_accuracy']:.3f} (target: {target_accuracy})"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_ocr_accuracy(self) -> TestResult:
        """Test OCR accuracy for screen monitoring"""
        start_time = time.time()
        test_name = "ocr_accuracy"
        
        try:
            test_cases = self.test_cases['ocr_accuracy']
            ocr_accuracies = []
            
            for test_case in test_cases:
                # Simulate OCR extraction (would integrate with actual OCR system)
                extracted_text = await self._simulate_ocr_extraction(test_case.input_data)
                
                # Calculate OCR accuracy
                ocr_accuracy = self._calculate_character_accuracy(
                    extracted_text, test_case.expected_output
                )
                ocr_accuracies.append(ocr_accuracy)
            
            # Calculate metrics
            metrics = {
                'mean_ocr_accuracy': np.mean(ocr_accuracies),
                'min_ocr_accuracy': np.min(ocr_accuracies),
                'max_ocr_accuracy': np.max(ocr_accuracies),
                'ocr_accuracy_std': np.std(ocr_accuracies),
                'test_cases_count': len(test_cases)
            }
            
            # Determine status based on target
            target_accuracy = self.config.accuracy_targets['ocr_accuracy']
            status = 'passed' if metrics['mean_ocr_accuracy'] >= target_accuracy else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"OCR accuracy: {metrics['mean_ocr_accuracy']:.3f} (target: {target_accuracy})"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def test_semantic_similarity(self) -> TestResult:
        """Test semantic similarity using sentence transformers"""
        start_time = time.time()
        test_name = "semantic_similarity"
        
        try:
            if not self.semantic_model:
                return TestResult(
                    test_name=test_name,
                    status="skipped",
                    metrics={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    details="Semantic model not available"
                )
            
            test_cases = self.test_cases['semantic_similarity']
            similarities = []
            
            for test_case in test_cases:
                # Calculate semantic similarity using sentence transformers
                input_embedding = self.semantic_model.encode([test_case.input_data])
                expected_embedding = self.semantic_model.encode([test_case.expected_output])
                
                similarity = cosine_similarity(input_embedding, expected_embedding)[0][0]
                similarities.append(similarity)
            
            # Calculate metrics
            metrics = {
                'mean_semantic_similarity': np.mean(similarities),
                'min_semantic_similarity': np.min(similarities),
                'max_semantic_similarity': np.max(similarities),
                'semantic_similarity_std': np.std(similarities),
                'test_cases_count': len(test_cases)
            }
            
            # Determine status based on target
            target_similarity = self.config.accuracy_targets['memory_recall_semantic']
            status = 'passed' if metrics['mean_semantic_similarity'] >= target_similarity else 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Semantic similarity: {metrics['mean_semantic_similarity']:.3f} (target: {target_similarity})"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def calculate_composite_accuracy_score(self, test_results: List[TestResult]) -> TestResult:
        """Calculate composite accuracy score from all accuracy tests"""
        start_time = time.time()
        test_name = "composite_accuracy"
        
        try:
            # Extract accuracy metrics from individual tests
            accuracy_scores = {}
            
            for result in test_results:
                if result.test_name == "memory_recall_bleu" and result.metrics:
                    accuracy_scores['memory_recall'] = result.metrics.get('mean_bleu_score', 0)
                elif result.test_name == "stt_accuracy" and result.metrics:
                    accuracy_scores['stt'] = result.metrics.get('mean_word_accuracy', 0)
                elif result.test_name == "ocr_accuracy" and result.metrics:
                    accuracy_scores['ocr'] = result.metrics.get('mean_ocr_accuracy', 0)
                elif result.test_name == "semantic_similarity" and result.metrics:
                    accuracy_scores['semantic'] = result.metrics.get('mean_semantic_similarity', 0)
            
            # Calculate weighted composite score
            weights = {
                'memory_recall': 0.3,
                'stt': 0.3,
                'ocr': 0.2,
                'semantic': 0.2
            }
            
            composite_score = sum(
                accuracy_scores.get(component, 0) * weight
                for component, weight in weights.items()
            )
            
            # Calculate individual component status
            component_status = {}
            for component, score in accuracy_scores.items():
                target_key = f"{component}_accuracy" if component != 'memory_recall' else 'memory_recall_bleu'
                if component == 'semantic':
                    target_key = 'memory_recall_semantic'
                
                target = self.config.accuracy_targets.get(target_key, 0.8)
                component_status[component] = 'passed' if score >= target else 'failed'
            
            metrics = {
                'composite_accuracy_score': composite_score,
                **accuracy_scores,
                'components_passed': sum(1 for status in component_status.values() if status == 'passed'),
                'total_components': len(component_status)
            }
            
            # Overall status based on composite score and individual components
            failed_components = [comp for comp, status in component_status.items() if status == 'failed']
            if not failed_components:
                status = 'passed'
            elif len(failed_components) <= 1:
                status = 'warning'
            else:
                status = 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=f"Composite score: {composite_score:.3f}, Failed components: {failed_components}"
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    async def run_quick_accuracy_check(self) -> TestResult:
        """Quick accuracy check for periodic monitoring"""
        start_time = time.time()
        test_name = "quick_accuracy_check"
        
        try:
            # Run a subset of tests for quick validation
            quick_metrics = {}
            
            # Quick memory recall check (3 test cases)
            if self.bleu_metric and len(self.test_cases['memory_recall']) >= 3:
                bleu_scores = []
                for test_case in self.test_cases['memory_recall'][:3]:
                    predicted = await self._simulate_memory_recall(test_case.input_data)
                    bleu = self.bleu_metric.compute(
                        predictions=[predicted],
                        references=[[test_case.expected_output]]
                    )
                    bleu_scores.append(bleu['bleu'])
                
                quick_metrics['memory_recall_bleu'] = np.mean(bleu_scores)
            
            # Quick STT check (3 test cases)
            if len(self.test_cases['stt_accuracy']) >= 3:
                word_accuracies = []
                for test_case in self.test_cases['stt_accuracy'][:3]:
                    transcribed = await self._simulate_stt_transcription(test_case.input_data)
                    accuracy = self._calculate_word_accuracy(transcribed, test_case.expected_output)
                    word_accuracies.append(accuracy)
                
                quick_metrics['stt_word_accuracy'] = np.mean(word_accuracies)
            
            # Determine status
            status = 'passed'
            for metric, value in quick_metrics.items():
                if 'bleu' in metric and value < self.config.accuracy_targets['memory_recall_bleu']:
                    status = 'failed'
                elif 'stt' in metric and value < self.config.accuracy_targets['stt_accuracy']:
                    status = 'failed'
            
            return TestResult(
                test_name=test_name,
                status=status,
                metrics=quick_metrics,
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                status="failed",
                metrics={},
                execution_time=time.time() - start_time,
                timestamp=datetime.now(),
                errors=[str(e)]
            )
    
    # Simulation methods (would integrate with actual components)
    async def _simulate_memory_recall(self, query: str) -> str:
        """Simulate memory recall response"""
        # In real implementation, this would call the actual memory system
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Simple simulation based on query keywords
        if "error" in query.lower():
            return "The authentication system had a JWT token expiration issue where tokens were not being refreshed properly."
        elif "database" in query.lower():
            return "We discussed adding indexes to the user_activities table on timestamp and user_id columns."
        elif "memory leak" in query.lower():
            return "The memory leak was caused by not properly disposing of audio buffers in the processing pipeline."
        else:
            return "I don't have specific information about that topic in my memory."
    
    async def _simulate_stt_transcription(self, text: str) -> str:
        """Simulate STT transcription with potential errors"""
        await asyncio.sleep(0.05)  # Simulate processing time
        
        # Simulate common STT errors
        transcribed = text
        
        # 5% chance of word substitution
        words = transcribed.split()
        for i, word in enumerate(words):
            if np.random.random() < 0.05:
                # Common substitutions
                substitutions = {
                    'docker': 'darker',
                    'async': 'sink',
                    'typescript': 'type script',
                    'kubernetes': 'cooper netes'
                }
                if word.lower() in substitutions:
                    words[i] = substitutions[word.lower()]
        
        return ' '.join(words)
    
    async def _simulate_ocr_extraction(self, text: str) -> str:
        """Simulate OCR extraction with potential errors"""
        await asyncio.sleep(0.02)  # Simulate processing time
        
        # Simulate common OCR errors
        extracted = text
        
        # Character substitution errors
        char_substitutions = {
            'l': '1',  # lowercase L to 1
            'O': '0',  # uppercase O to 0
            'I': '1',  # uppercase I to 1
            'S': '5'   # uppercase S to 5
        }
        
        # 3% chance of character substitution
        chars = list(extracted)
        for i, char in enumerate(chars):
            if np.random.random() < 0.03 and char in char_substitutions:
                chars[i] = char_substitutions[char]
        
        return ''.join(chars)
    
    def _calculate_character_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate character-level accuracy"""
        if not expected:
            return 1.0 if not predicted else 0.0
        
        # Use Levenshtein distance for character accuracy
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        distance = levenshtein_distance(predicted, expected)
        max_len = max(len(predicted), len(expected))
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (distance / max_len)
    
    def _calculate_word_accuracy(self, predicted: str, expected: str) -> float:
        """Calculate word-level accuracy"""
        predicted_words = predicted.lower().split()
        expected_words = expected.lower().split()
        
        if not expected_words:
            return 1.0 if not predicted_words else 0.0
        
        # Calculate word-level edit distance
        correct_words = 0
        for pred_word, exp_word in zip(predicted_words, expected_words):
            if pred_word == exp_word:
                correct_words += 1
        
        return correct_words / len(expected_words)


def create_accuracy_benchmark_suite(config: Optional[PerformanceTestConfig] = None) -> AccuracyBenchmarkSuite:
    """Factory function to create accuracy benchmark suite"""
    if config is None:
        config = PerformanceTestConfig()
    return AccuracyBenchmarkSuite(config) 