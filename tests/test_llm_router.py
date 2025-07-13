import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from assistant.llm_router import (
    QueryClassifier, 
    QueryComplexity, 
    QueryClassification,
    LLMRouter
)
from assistant.router_config import RouterConfig, ModelConfig

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

class TestQueryClassifier:
    """Test cases for the QueryClassifier"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = QueryClassifier()
    
    def test_simple_queries(self):
        """Test that simple queries are classified correctly"""
        simple_queries = [
            "Hello",
            "Hi there",
            "What time is it?",
            "Thanks",
            "Yes",
            "No",
            "Stop",
            "Play music",
            "Volume up",
            "Good morning",
            "Bye"
        ]
        
        for query in simple_queries:
            classification = self.classifier.classify_query(query)
            assert classification.complexity == QueryComplexity.SIMPLE, f"Query '{query}' should be simple"
            assert classification.confidence > 0.5, f"Query '{query}' should have reasonable confidence"
    
    def test_complex_queries(self):
        """Test that complex queries are classified correctly"""
        complex_queries = [
            "How do I implement a microservices architecture for my application?",
            "Explain the best practices for database design and optimization",
            "What are the pros and cons of using React vs Vue.js for frontend development?",
            "Can you analyze the performance implications of using Docker containers?",
            "I need a detailed explanation of machine learning algorithms",
            "How should I design a scalable API architecture?",
            "What are the security considerations for implementing OAuth2?",
            "Explain step by step how to migrate from SQL to NoSQL database",
            "Compare and contrast different CI/CD deployment strategies",
            "What are the best practices for troubleshooting production issues?"
        ]
        
        for query in complex_queries:
            classification = self.classifier.classify_query(query)
            assert classification.complexity == QueryComplexity.COMPLEX, f"Query '{query}' should be complex"
            assert classification.confidence > 0.6, f"Query '{query}' should have high confidence"
    
    def test_length_analysis(self):
        """Test length-based classification"""
        # Very short query
        short_query = "Hi"
        short_classification = self.classifier.classify_query(short_query)
        assert short_classification.complexity == QueryComplexity.SIMPLE
        
        # Very long query
        long_query = "This is a very long query that contains many words and should be classified as complex based on length alone even without any technical keywords or patterns because it clearly requires more processing"
        long_classification = self.classifier.classify_query(long_query)
        assert long_classification.complexity == QueryComplexity.COMPLEX
    
    def test_keyword_analysis(self):
        """Test keyword-based classification"""
        # Query with complex keywords
        tech_query = "How to implement microservices architecture with Docker and Kubernetes"
        tech_classification = self.classifier.classify_query(tech_query)
        assert tech_classification.complexity == QueryComplexity.COMPLEX
        assert tech_classification.factors['complex_keywords'] >= 2
        
        # Query with simple keywords
        simple_query = "What time is it please"
        simple_classification = self.classifier.classify_query(simple_query)
        assert simple_classification.complexity == QueryComplexity.SIMPLE
        assert simple_classification.factors['simple_keywords'] >= 1
    
    def test_pattern_matching(self):
        """Test pattern-based classification"""
        pattern_queries = [
            "How do I implement authentication in my app?",
            "What is the best way to optimize database performance?",
            "Explain how microservices work",
            "Step by step guide to deployment",
            "Detailed explanation of the architecture"
        ]
        
        for query in pattern_queries:
            classification = self.classifier.classify_query(query)
            assert classification.complexity == QueryComplexity.COMPLEX
            assert classification.factors['pattern_matches'] > 0
    
    def test_context_analysis(self):
        """Test context-based classification"""
        # Without context
        query = "Tell me more"
        classification_no_context = self.classifier.classify_query(query)
        
        # With simple context
        simple_context = ["Hello", "How are you?"]
        classification_simple_context = self.classifier.classify_query(query, simple_context)
        
        # With complex context
        complex_context = [
            "How do I implement microservices architecture?",
            "You should consider using Docker containers and Kubernetes for orchestration..."
        ]
        classification_complex_context = self.classifier.classify_query(query, complex_context)
        
        # Complex context should increase likelihood of complex classification
        assert classification_complex_context.factors['context_indicators'] > classification_simple_context.factors['context_indicators']
    
    def test_borderline_cases(self):
        """Test borderline cases that could go either way"""
        borderline_queries = [
            "How does this work?",
            "Can you help me with this?",
            "What should I do next?",
            "Is this the right approach?"
        ]
        
        for query in borderline_queries:
            classification = self.classifier.classify_query(query)
            # Borderline cases should default to simple for faster response
            assert classification.complexity == QueryComplexity.SIMPLE
            assert 0.4 <= classification.confidence <= 0.9
    
    def test_reasoning_generation(self):
        """Test that classification reasoning is generated properly"""
        query = "How do I implement microservices architecture using Docker and Kubernetes?"
        classification = self.classifier.classify_query(query)
        
        assert classification.reasoning is not None
        assert len(classification.reasoning) > 0
        assert "complex" in classification.reasoning.lower()
        assert "score:" in classification.reasoning
    
    def test_factors_collection(self):
        """Test that classification factors are collected properly"""
        query = "How do I implement microservices architecture?"
        classification = self.classifier.classify_query(query)
        
        assert 'length' in classification.factors
        assert 'complex_keywords' in classification.factors
        assert 'simple_keywords' in classification.factors
        assert 'pattern_matches' in classification.factors
        assert 'context_indicators' in classification.factors
        
        assert classification.factors['length'] == len(query)
        assert classification.factors['complex_keywords'] >= 1  # Should find 'implement', 'microservices', 'architecture'
        assert classification.factors['pattern_matches'] >= 1   # Should match "how do i implement" pattern

class TestLLMRouter:
    """Test cases for the LLMRouter"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Create test configuration
        test_config = RouterConfig(
            openrouter_api_key="test-api-key",
            enable_rate_limiting=False,  # Disable for testing
            log_requests=False,
            log_responses=False,
            models={
                'fast': ModelConfig(
                    id='openai/gpt-4o-mini',
                    name='GPT-4o-mini',
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=10.0,
                    cost_per_1k_tokens=0.0015
                ),
                'deep': ModelConfig(
                    id='openai/gpt-4o',
                    name='GPT-4o',
                    max_tokens=2000,
                    temperature=0.8,
                    timeout=60.0,
                    cost_per_1k_tokens=0.015
                )
            }
        )
        self.router = LLMRouter(test_config)
    
    def test_router_initialization(self):
        """Test router initialization"""
        assert self.router.classifier is not None
        assert self.router.openrouter_client is not None
        assert len(self.router.config.models) == 2
        assert 'fast' in self.router.config.models
        assert 'deep' in self.router.config.models
        assert self.router.conversation_history == []
    
    def test_model_configuration(self):
        """Test model configuration"""
        fast_model = self.router.config.models['fast']
        deep_model = self.router.config.models['deep']
        
        # Check required fields
        for model in [fast_model, deep_model]:
            assert model.id is not None
            assert model.name is not None
            assert model.max_tokens > 0
            assert model.temperature >= 0
            assert model.timeout > 0
            assert model.cost_per_1k_tokens >= 0
        
        # Fast model should have shorter timeout
        assert fast_model.timeout < deep_model.timeout
        
        # Deep model should have higher cost
        assert deep_model.cost_per_1k_tokens > fast_model.cost_per_1k_tokens
    
    @pytest.mark.asyncio
    async def test_route_query_simple(self):
        """Test routing a simple query"""
        # Mock the OpenAI client's chat.completions.create method directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        
        with patch.object(self.router.openrouter_client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
            result = await self.router.route_query("Hello")
        
        assert result['success'] is True
        assert result['model_type'] == 'fast'
        assert result['model_used'] == 'GPT-4o-mini'
        assert result['classification'].complexity == QueryComplexity.SIMPLE
        assert 'response' in result
        assert 'timing' in result
    
    @pytest.mark.asyncio
    async def test_route_query_complex(self):
        """Test routing a complex query"""
        # Mock the OpenAI client's chat.completions.create method directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Here's a detailed explanation of microservices architecture..."
        
        with patch.object(self.router.openrouter_client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
            result = await self.router.route_query("How do I implement microservices architecture?")
        
        assert result['success'] is True
        assert result['model_type'] == 'deep'
        assert result['model_used'] == 'GPT-4o'
        assert result['classification'].complexity == QueryComplexity.COMPLEX
        assert 'response' in result
        assert 'timing' in result
    
    @pytest.mark.asyncio
    async def test_conversation_history_management(self):
        """Test that conversation history is managed correctly"""
        # Mock the OpenAI client's chat.completions.create method directly
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        
        with patch.object(self.router.openrouter_client.chat.completions, 'create', new=AsyncMock(return_value=mock_response)):
            # Initial state
            assert len(self.router.conversation_history) == 0
            
            # First query
            await self.router.route_query("Hello")
            assert len(self.router.conversation_history) == 2  # Query + response
            
            # Second query
            await self.router.route_query("How are you?")
            assert len(self.router.conversation_history) == 4  # 2 queries + 2 responses
            
            # Test history trimming (simulate reaching max length)
            self.router.max_context_length = 4
            await self.router.route_query("Another query")
            assert len(self.router.conversation_history) == 4  # Should be trimmed
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in router"""
        # Mock an exception
        with patch.object(self.router.openrouter_client.chat.completions, 'create', new=AsyncMock(side_effect=Exception("API Error"))):
            result = await self.router.route_query("Test query")
        
        assert result['success'] is False
        assert 'error' in result
        assert "API Error" in result['error']
        assert result['response'].startswith("I apologize")
    
    @pytest.mark.asyncio
    async def test_system_prompt_usage(self):
        """Test that system prompts are used correctly"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with system prompt"
        
        mock_create = AsyncMock(return_value=mock_response)
        
        with patch.object(self.router.openrouter_client.chat.completions, 'create', new=mock_create):
            system_prompt = "You are a helpful assistant specializing in technical topics."
            await self.router.route_query("Hello", system_prompt=system_prompt)
            
            # Check that the system prompt was included in the API call
            call_args = mock_create.call_args
            messages = call_args[1]['messages']
            
            assert messages[0]['role'] == 'system'
            assert messages[0]['content'] == system_prompt
    
    def test_get_stats(self):
        """Test router statistics"""
        stats = self.router.get_stats()
        
        assert 'conversation_length' in stats
        assert 'models_configured' in stats
        assert 'classifier_config' in stats
        assert 'performance' in stats
        assert 'configuration' in stats
        assert 'models' in stats
        
        assert stats['conversation_length'] == 0
        assert stats['models_configured'] == 2
        assert isinstance(stats['classifier_config'], dict)
        assert isinstance(stats['performance'], dict)
        assert isinstance(stats['configuration'], dict)
        assert isinstance(stats['models'], dict)

class TestIntegration:
    """Integration tests for the complete routing system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.classifier = QueryClassifier()
        # Create test configuration
        test_config = RouterConfig(
            openrouter_api_key="test-api-key",
            enable_rate_limiting=False,  # Disable for testing
            log_requests=False,
            log_responses=False
        )
        self.router = LLMRouter(test_config)
    
    def test_end_to_end_classification(self):
        """Test end-to-end classification with various query types"""
        test_cases = [
            ("Hi", QueryComplexity.SIMPLE),
            ("What time is it?", QueryComplexity.SIMPLE),
            ("Thanks for your help", QueryComplexity.SIMPLE),
            ("How do I implement OAuth2 authentication?", QueryComplexity.COMPLEX),
            ("Explain the microservices architecture pattern", QueryComplexity.COMPLEX),
            ("What are the best practices for database optimization?", QueryComplexity.COMPLEX),
        ]
        
        for query, expected_complexity in test_cases:
            classification = self.classifier.classify_query(query)
            assert classification.complexity == expected_complexity, f"Query '{query}' classification failed"
    
    def test_classification_consistency(self):
        """Test that the same query gets consistent classification"""
        query = "How do I implement microservices architecture?"
        
        # Classify the same query multiple times
        classifications = [
            self.classifier.classify_query(query) for _ in range(5)
        ]
        
        # All should have the same complexity
        complexities = [c.complexity for c in classifications]
        assert all(c == complexities[0] for c in complexities), "Classifications should be consistent"
    
    def test_performance_requirements(self):
        """Test that classification meets performance requirements"""
        import time
        
        queries = [
            "Hello",
            "How are you?",
            "What time is it?",
            "How do I implement microservices architecture?",
            "Explain the best practices for database optimization"
        ]
        
        start_time = time.time()
        for query in queries:
            self.classifier.classify_query(query)
        end_time = time.time()
        
        # Classification should be very fast (< 0.1s for 5 queries)
        total_time = end_time - start_time
        assert total_time < 0.1, f"Classification took too long: {total_time:.3f}s"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 