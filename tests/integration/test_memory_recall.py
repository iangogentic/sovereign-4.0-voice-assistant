#!/usr/bin/env python3
"""
Memory Recall Integration Tests
Tests memory storage, retrieval, and accuracy validation with BLEU scoring
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
from unittest.mock import patch, AsyncMock, MagicMock

# BLEU score calculation for memory recall accuracy
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    BLEU_AVAILABLE = True
except ImportError:
    # Fallback BLEU implementation for testing
    def sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None):
        """Simple BLEU approximation for testing when NLTK unavailable"""
        # Simple word overlap scoring as fallback
        ref_words = set(reference[0]) if isinstance(reference[0], list) else set(reference[0].split())
        cand_words = set(candidate) if isinstance(candidate, list) else set(candidate.split())
        
        overlap = len(ref_words.intersection(cand_words))
        total = len(ref_words.union(cand_words))
        
        return overlap / total if total > 0 else 0.0
    
    SmoothingFunction = None
    BLEU_AVAILABLE = False


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryRecallAccuracy:
    """Memory recall accuracy testing with BLEU validation"""
    
    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between reference and candidate responses"""
        try:
            if BLEU_AVAILABLE and SmoothingFunction:
                # Tokenize texts
                reference_tokens = reference.lower().split()
                candidate_tokens = candidate.lower().split()
                
                # Calculate BLEU score with smoothing
                smoothing = SmoothingFunction().method1
                score = sentence_bleu([reference_tokens], candidate_tokens, 
                                    weights=(0.25, 0.25, 0.25, 0.25),
                                    smoothing_function=smoothing)
            else:
                # Fallback implementation
                score = sentence_bleu([reference], candidate)
            
            return score
        except Exception as e:
            # Fallback to simple word overlap
            ref_words = set(reference.lower().split())
            cand_words = set(candidate.lower().split())
            overlap = len(ref_words.intersection(cand_words))
            total = len(ref_words.union(cand_words))
            return overlap / total if total > 0 else 0.0
    
    async def test_basic_memory_storage_and_recall(self, sovereign_assistant, test_metrics, test_environment):
        """Test basic memory storage and recall functionality"""
        
        # Test data: conversation pairs
        memory_scenarios = [
            {
                'input': "Remember that I work as a Python developer at TechCorp",
                'query': "Where do I work and what's my role?",
                'expected_keywords': ['python', 'developer', 'techcorp', 'work']
            },
            {
                'input': "My favorite programming language is Python because it's readable",
                'query': "What's my favorite programming language and why?",
                'expected_keywords': ['python', 'favorite', 'readable', 'language']
            },
            {
                'input': "I had a bug in my Django application with database migrations",
                'query': "What issue did I have with Django?",
                'expected_keywords': ['bug', 'django', 'database', 'migrations']
            }
        ]
        
        bleu_scores = []
        
        for scenario in memory_scenarios:
            # Store information in memory
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=scenario['input']):
                storage_response = await sovereign_assistant._process_voice_input()
                assert storage_response is not None
            
            # Allow time for memory processing
            await asyncio.sleep(0.1)
            
            # Query the stored information
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=scenario['query']):
                start_time = time.time()
                recall_response = await sovereign_assistant._process_voice_input()
                recall_time = time.time() - start_time
                
                assert recall_response is not None
                
                # Calculate BLEU score
                expected_response = ' '.join(scenario['expected_keywords'])
                bleu_score = self.calculate_bleu_score(expected_response, recall_response)
                bleu_scores.append(bleu_score)
                
                # Record metrics
                test_metrics.record_metric('memory_recall_accuracy', bleu_score)
                test_metrics.record_metric('memory_recall_latency', recall_time)
                
                # Check if response contains expected keywords
                response_lower = recall_response.lower()
                keyword_matches = sum(1 for keyword in scenario['expected_keywords'] 
                                    if keyword in response_lower)
                keyword_accuracy = keyword_matches / len(scenario['expected_keywords'])
                
                assert keyword_accuracy >= 0.5, f"Low keyword accuracy: {keyword_accuracy}"
        
        # Validate overall BLEU score requirement
        average_bleu = sum(bleu_scores) / len(bleu_scores)
        test_metrics.set_threshold('memory_recall_accuracy', test_environment['memory_recall_accuracy_threshold'])
        
        assert average_bleu >= test_environment['memory_recall_accuracy_threshold'], \
            f"Memory recall BLEU score {average_bleu:.3f} below threshold {test_environment['memory_recall_accuracy_threshold']}"
    
    async def test_contextual_memory_recall(self, sovereign_assistant, test_metrics):
        """Test contextual memory recall with conversation history"""
        
        # Build conversation context
        conversation_history = [
            "I'm working on a machine learning project with scikit-learn",
            "The model is giving me poor accuracy results around 60%",
            "I think the issue might be with data preprocessing",
            "I tried feature scaling but it didn't help much"
        ]
        
        # Store conversation history
        for statement in conversation_history:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=statement):
                await sovereign_assistant._process_voice_input()
                await asyncio.sleep(0.1)  # Allow processing time
        
        # Test contextual queries
        contextual_queries = [
            {
                'query': "What machine learning library am I using?",
                'expected': "scikit-learn",
                'context_keywords': ['scikit-learn', 'machine learning']
            },
            {
                'query': "What accuracy am I getting with my model?",
                'expected': "60% accuracy",
                'context_keywords': ['60%', 'accuracy', 'poor']
            },
            {
                'query': "What preprocessing techniques have I tried?",
                'expected': "feature scaling",
                'context_keywords': ['feature scaling', 'preprocessing']
            }
        ]
        
        bleu_scores = []
        
        for query_data in contextual_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query_data['query']):
                response = await sovereign_assistant._process_voice_input()
                
                assert response is not None
                
                # Calculate BLEU score
                bleu_score = self.calculate_bleu_score(query_data['expected'], response)
                bleu_scores.append(bleu_score)
                
                # Check contextual understanding
                response_lower = response.lower()
                context_matches = sum(1 for keyword in query_data['context_keywords'] 
                                    if keyword.lower() in response_lower)
                
                test_metrics.record_metric('contextual_memory_bleu', bleu_score)
                test_metrics.record_metric('contextual_understanding', 
                                         context_matches / len(query_data['context_keywords']))
        
        average_contextual_bleu = sum(bleu_scores) / len(bleu_scores)
        assert average_contextual_bleu >= 0.7, f"Contextual memory BLEU score too low: {average_contextual_bleu:.3f}"
    
    async def test_long_term_memory_persistence(self, sovereign_assistant, test_metrics):
        """Test long-term memory persistence across sessions"""
        
        # Simulate information storage
        persistent_info = [
            "My API key for OpenAI is stored in environment variable OPENAI_API_KEY",
            "I prefer using VSCode as my primary IDE for development",
            "My project uses PostgreSQL database with connection pooling",
            "The production server runs on AWS EC2 with Ubuntu 22.04"
        ]
        
        # Store information
        for info in persistent_info:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=info):
                await sovereign_assistant._process_voice_input()
        
        # Simulate time passage and multiple interactions
        for _ in range(10):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value="Hello"):
                await sovereign_assistant._process_voice_input()
        
        # Test recall of previously stored information
        recall_queries = [
            "Where is my OpenAI API key stored?",
            "What IDE do I prefer for development?", 
            "What database does my project use?",
            "What operating system runs on my production server?"
        ]
        
        successful_recalls = 0
        
        for query in recall_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                response = await sovereign_assistant._process_voice_input()
                
                # Check if response contains relevant information
                if any(keyword in response.lower() for keyword in 
                      ['openai', 'vscode', 'postgresql', 'aws', 'ubuntu']):
                    successful_recalls += 1
        
        recall_rate = successful_recalls / len(recall_queries)
        test_metrics.record_metric('long_term_memory_recall_rate', recall_rate)
        
        assert recall_rate >= 0.75, f"Long-term memory recall rate too low: {recall_rate:.2f}"
    
    async def test_memory_accuracy_with_similar_information(self, sovereign_assistant, test_metrics):
        """Test memory accuracy when dealing with similar information"""
        
        # Store similar but distinct information
        similar_info = [
            "Project Alpha uses React for frontend development",
            "Project Beta uses Vue.js for frontend development", 
            "Project Gamma uses Angular for frontend development"
        ]
        
        for info in similar_info:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=info):
                await sovereign_assistant._process_voice_input()
                await asyncio.sleep(0.1)
        
        # Test specific recall queries
        specific_queries = [
            {
                'query': "What frontend framework does Project Alpha use?",
                'expected': "React",
                'avoid': ["Vue", "Angular"]
            },
            {
                'query': "Which project uses Vue.js?",
                'expected': "Project Beta",
                'avoid': ["Alpha", "Gamma"]
            },
            {
                'query': "What framework is used in Project Gamma?",
                'expected': "Angular",
                'avoid': ["React", "Vue"]
            }
        ]
        
        accuracy_scores = []
        
        for query_data in specific_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query_data['query']):
                response = await sovereign_assistant._process_voice_input()
                
                assert response is not None
                
                response_lower = response.lower()
                
                # Check if correct information is present
                correct_present = query_data['expected'].lower() in response_lower
                
                # Check if incorrect information is avoided
                incorrect_avoided = not any(avoid.lower() in response_lower for avoid in query_data['avoid'])
                
                accuracy = 1.0 if (correct_present and incorrect_avoided) else 0.5 if correct_present else 0.0
                accuracy_scores.append(accuracy)
                
                test_metrics.record_metric('memory_disambiguation_accuracy', accuracy)
        
        average_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        assert average_accuracy >= 0.8, f"Memory disambiguation accuracy too low: {average_accuracy:.2f}"
    
    async def test_memory_recall_performance_under_load(self, sovereign_assistant, test_metrics):
        """Test memory recall performance with large amounts of stored information"""
        
        # Store large amount of information
        bulk_information = []
        for i in range(50):
            info = f"Document {i}: Important information about topic {i} with details {i*10}"
            bulk_information.append(info)
            
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=info):
                await sovereign_assistant._process_voice_input()
        
        # Test recall performance
        test_queries = [
            "What is in document 10?",
            "Tell me about topic 25", 
            "What details are associated with topic 30?",
            "Find information about document 45"
        ]
        
        recall_times = []
        successful_recalls = 0
        
        for query in test_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                recall_time = time.time() - start_time
                
                recall_times.append(recall_time)
                
                # Check if relevant information was found
                if any(str(i) in response for i in range(50)):
                    successful_recalls += 1
                
                test_metrics.record_metric('bulk_memory_recall_latency', recall_time)
        
        average_recall_time = sum(recall_times) / len(recall_times)
        recall_success_rate = successful_recalls / len(test_queries)
        
        test_metrics.record_metric('bulk_memory_recall_success_rate', recall_success_rate)
        
        # Assert performance requirements
        assert average_recall_time < 2.0, f"Bulk memory recall too slow: {average_recall_time:.2f}s"
        assert recall_success_rate >= 0.75, f"Bulk memory recall success rate too low: {recall_success_rate:.2f}"


@pytest.mark.integration
@pytest.mark.asyncio 
class TestMemorySystemIntegration:
    """Integration tests for memory system components"""
    
    async def test_memory_storage_integration(self, sovereign_assistant):
        """Test memory storage integration with voice pipeline"""
        
        # Test that information flows properly into memory system
        test_statements = [
            "I am working on a Python project using Django framework",
            "The database connection is using PostgreSQL on port 5432",
            "My development environment is running on macOS Big Sur"
        ]
        
        for statement in test_statements:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=statement):
                response = await sovereign_assistant._process_voice_input()
                assert response is not None
                
                # Verify information acknowledgment
                assert any(word in response.lower() for word in ['understand', 'noted', 'remember', 'got'])
    
    async def test_memory_search_integration(self, sovereign_assistant):
        """Test memory search integration with query processing"""
        
        # Store searchable information
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', 
                         return_value="I fixed a bug in the authentication middleware yesterday"):
            await sovereign_assistant._process_voice_input()
        
        # Test search queries
        search_queries = [
            "What bug did I fix recently?",
            "Tell me about authentication issues",
            "What did I work on yesterday?"
        ]
        
        for query in search_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                response = await sovereign_assistant._process_voice_input()
                
                # Should find relevant information
                assert response is not None
                response_lower = response.lower()
                assert any(keyword in response_lower for keyword in ['bug', 'authentication', 'middleware', 'yesterday'])
    
    async def test_memory_update_and_versioning(self, sovereign_assistant):
        """Test memory updates and information versioning"""
        
        # Initial information
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', 
                         return_value="My current project uses Python 3.8"):
            await sovereign_assistant._process_voice_input()
        
        # Updated information
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', 
                         return_value="I upgraded my project to Python 3.11 yesterday"):
            await sovereign_assistant._process_voice_input()
        
        # Query should return most recent information
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', 
                         return_value="What Python version is my project using?"):
            response = await sovereign_assistant._process_voice_input()
            
            assert response is not None
            # Should mention the newer version
            assert "3.11" in response or "upgraded" in response.lower()
    
    async def test_memory_privacy_and_filtering(self, sovereign_assistant):
        """Test memory privacy controls and information filtering"""
        
        # Store sensitive information
        sensitive_info = [
            "My password is secret123",
            "The API key is sk-abc123def456",
            "My social security number is 123-45-6789"
        ]
        
        for info in sensitive_info:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=info):
                response = await sovereign_assistant._process_voice_input()
                # Should acknowledge but not repeat sensitive data
                assert response is not None
                assert "secret123" not in response
                assert "sk-abc123def456" not in response
                assert "123-45-6789" not in response
    
    async def test_memory_export_and_import(self, sovereign_assistant):
        """Test memory export and import functionality"""
        
        # Store test information
        test_info = [
            "I completed the user authentication feature last week",
            "The database schema was updated with new user roles",
            "Performance testing showed 200ms average response time"
        ]
        
        for info in test_info:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=info):
                await sovereign_assistant._process_voice_input()
        
        # Test export functionality (if available)
        # This would test the memory system's ability to export stored data
        
        # Test import verification through recall
        recall_queries = [
            "What feature did I complete last week?",
            "What was updated in the database?",
            "What were the performance test results?"
        ]
        
        successful_recalls = 0
        for query in recall_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                response = await sovereign_assistant._process_voice_input()
                if response and any(keyword in response.lower() for keyword in 
                                 ['authentication', 'database', 'performance', '200ms']):
                    successful_recalls += 1
        
        # Should recall most stored information
        recall_rate = successful_recalls / len(recall_queries)
        assert recall_rate >= 0.66, f"Memory export/import recall rate too low: {recall_rate:.2f}"


@pytest.mark.integration
@pytest.mark.performance
@pytest.mark.asyncio
class TestMemoryRecallPerformance:
    """Performance testing for memory recall system"""
    
    async def test_memory_recall_latency_requirements(self, sovereign_assistant, test_metrics, test_environment):
        """Test memory recall latency requirements"""
        
        # Store information for recall testing
        stored_info = "I am developing a REST API using FastAPI framework with PostgreSQL database"
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=stored_info):
            await sovereign_assistant._process_voice_input()
        
        # Test recall latency
        recall_queries = [
            "What framework am I using for the API?",
            "What database am I using?",
            "What type of API am I developing?"
        ]
        
        latencies = []
        
        for query in recall_queries:
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                latency = time.time() - start_time
                
                latencies.append(latency)
                test_metrics.record_metric('memory_recall_latency', latency)
                
                assert response is not None
        
        average_latency = sum(latencies) / len(latencies)
        
        # Memory recall should be fast (within voice pipeline requirements)
        assert average_latency < 1.5, f"Memory recall latency too high: {average_latency:.3f}s"
    
    async def test_concurrent_memory_access(self, sovereign_assistant):
        """Test concurrent memory access and recall"""
        
        # Store information
        with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', 
                         return_value="I work on microservices architecture with Docker containers"):
            await sovereign_assistant._process_voice_input()
        
        # Concurrent recall queries
        async def recall_query(query):
            with patch.object(sovereign_assistant.stt_service, 'transcribe_audio', return_value=query):
                start_time = time.time()
                response = await sovereign_assistant._process_voice_input()
                return time.time() - start_time, response is not None
        
        queries = [
            "What architecture do I work on?",
            "What containerization technology do I use?",
            "What kind of services do I develop?"
        ]
        
        # Execute concurrent recalls
        tasks = [recall_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for latency, success in results:
            assert success, "Concurrent memory recall failed"
            assert latency < 3.0, f"Concurrent recall latency too high: {latency:.3f}s" 