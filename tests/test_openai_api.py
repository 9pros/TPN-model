"""
Tests for OpenAI chat completions API format.
"""

import pytest
from tpn_engine.openai_api import OpenAIChatCompletions, ChatMessage, ChatCompletionRequest


class TestChatMessage:
    """Test chat message."""
    
    def test_message_creation(self):
        """Test creating a chat message."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = ChatMessage(role="assistant", content="Hi there")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Hi there"
    
    def test_message_from_dict(self):
        """Test creating message from dictionary."""
        d = {"role": "user", "content": "Hello"}
        msg = ChatMessage.from_dict(d)
        assert msg.role == "user"
        assert msg.content == "Hello"


class TestChatCompletionRequest:
    """Test chat completion request."""
    
    def test_request_creation(self):
        """Test creating a chat completion request."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(
            model="tpn-model",
            messages=messages,
            max_tokens=100,
            temperature=0.7,
        )
        
        assert request.model == "tpn-model"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.temperature == 0.7
    
    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        messages = [ChatMessage(role="user", content="Hello")]
        request = ChatCompletionRequest(
            model="tpn-model",
            messages=messages,
        )
        
        d = request.to_dict()
        assert d["model"] == "tpn-model"
        assert len(d["messages"]) == 1
        assert d["messages"][0]["role"] == "user"


class TestOpenAIChatCompletions:
    """Test OpenAI chat completions API."""
    
    def test_api_creation(self):
        """Test creating OpenAI chat completions API."""
        api = OpenAIChatCompletions()
        assert api is not None
    
    def test_create_completion(self):
        """Test creating a chat completion."""
        api = OpenAIChatCompletions()
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = api.create_completion(
            model="tpn-model",
            messages=messages,
            max_tokens=50,
        )
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
    
    def test_create_completion_with_tools(self):
        """Test creating a chat completion with tools."""
        api = OpenAIChatCompletions()
        
        # Register a tool
        def add(a: int, b: int) -> int:
            return a + b
        
        api.register_tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
        )
        
        messages = [ChatMessage(role="user", content="What is 2+2?")]
        response = api.create_completion(
            model="tpn-model",
            messages=messages,
            tools=api.get_tools(),
            max_tokens=100,
        )
        
        assert "choices" in response
        assert len(response["choices"]) > 0
    
    def test_create_completion_with_tool_call(self):
        """Test creating a chat completion that uses a tool."""
        api = OpenAIChatCompletions()
        
        # Register a tool
        def add(a: int, b: int) -> int:
            return a + b
        
        api.register_tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            }
        )
        
        # Simulate a conversation with tool calls
        messages = [
            ChatMessage(role="user", content="What is 2+2?"),
            ChatMessage(role="assistant", content="", tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": '{"a": 2, "b": 2}'}
            }]),
        ]
        
        response = api.create_completion(
            model="tpn-model",
            messages=messages,
            tools=api.get_tools(),
            max_tokens=100,
        )
        
        assert "choices" in response
    
    def test_response_format(self):
        """Test that response is in OpenAI format."""
        api = OpenAIChatCompletions()
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = api.create_completion(
            model="tpn-model",
            messages=messages,
        )
        
        # Check OpenAI response format
        assert "id" in response
        assert "object" in response
        assert response["object"] == "chat.completion"
        assert "created" in response
        assert "model" in response
        assert "choices" in response
        assert "usage" in response
        
        # Check choice format
        choice = response["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice
        
        # Check message format
        message = choice["message"]
        assert "role" in message
        assert "content" in message
    
    def test_streaming_response(self):
        """Test streaming response format."""
        api = OpenAIChatCompletions()
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = api.create_completion(
            model="tpn-model",
            messages=messages,
            stream=True,
        )
        
        # Streaming response should be a generator
        assert hasattr(response, '__iter__')
        
        # Collect chunks
        chunks = list(response)
        assert len(chunks) > 0
        
        # Check chunk format
        for chunk in chunks:
            assert "choices" in chunk
            assert len(chunk["choices"]) > 0
