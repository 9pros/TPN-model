"""
OpenAI chat completions API format for TPN.

Provides OpenAI-compatible chat completions API with:
- Chat message format
- Tool calling support
- Streaming responses
- OpenAI response format
"""

import json
import time
import uuid
from typing import Dict, List, Optional, Any, Generator, Union
from dataclasses import dataclass, field

from .tool_calling import Tool, ToolCall, ToolRegistry, ToolExecutor


@dataclass
class ChatMessage:
    """A chat message."""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create from dictionary."""
        return cls(
            role=data.get("role", "user"),
            content=data.get("content"),
            name=data.get("name"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ChatCompletionRequest:
    """A chat completion request."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n,
            "stream": self.stream,
        }
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.stop is not None:
            d["stop"] = self.stop
        if self.tools is not None:
            d["tools"] = self.tools
        if self.tool_choice is not None:
            d["tool_choice"] = self.tool_choice
        return d


class OpenAIChatCompletions:
    """
    OpenAI-compatible chat completions API.
    
    Provides a drop-in replacement for OpenAI's chat completions API
    that uses the TPN engine for inference.
    """
    
    def __init__(self, model=None):
        """
        Initialize OpenAI chat completions API.
        
        Args:
            model: TPN model to use for inference.
        """
        self.model = model
        self.tool_registry = ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
    
    def register_tool(self, name: str, description: str, function, parameters: Dict[str, Any]) -> None:
        """Register a tool."""
        tool = Tool(
            name=name,
            description=description,
            function=function,
            parameters=parameters,
        )
        self.tool_registry.register(tool)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get registered tools in OpenAI format."""
        return self.tool_registry.to_openai_tools()
    
    def create_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            model: Model name.
            messages: List of chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            n: Number of completions to generate.
            stream: Whether to stream the response.
            stop: Stop sequence(s).
            tools: Available tools.
            tool_choice: Tool choice strategy.
        
        Returns:
            OpenAI-formatted chat completion response.
        """
        if stream:
            return self._create_completion_stream(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=n,
                stop=stop,
                tools=tools,
                tool_choice=tool_choice,
            )
        
        # Generate response
        response_text = self._generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )
        
        # Build OpenAI response
        return self._build_response(
            model=model,
            response_text=response_text,
            messages=messages,
            max_tokens=max_tokens,
        )
    
    def _generate_response(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate response text."""
        # Build prompt from messages
        prompt = self._build_prompt(messages)
        
        # If we have a TPN model, use it
        if self.model is not None:
            # Tokenize prompt (simple whitespace tokenization for now)
            tokens = prompt.split()
            
            # Run inference
            output = self.model.forward([float(len(tokens))])
            
            # Convert output to text (placeholder)
            response = self._output_to_text(output)
        else:
            # Placeholder response
            response = "This is a placeholder response from the TPN engine."
        
        return response
    
    def _build_prompt(self, messages: List[ChatMessage]) -> str:
        """Build prompt from messages."""
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                parts.append(f"Tool: {msg.content}")
        return "\n".join(parts)
    
    def _output_to_text(self, output: List[float]) -> str:
        """Convert model output to text."""
        # Placeholder: convert first few floats to characters
        chars = []
        for val in output[:20]:
            # Map float to ASCII character
            char_code = int(abs(val) * 100) % 26 + 97  # a-z
            chars.append(chr(char_code))
        return "".join(chars)
    
    def _build_response(
        self,
        model: str,
        response_text: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build OpenAI-formatted response."""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.content or "") for m in messages),
                "completion_tokens": len(response_text),
                "total_tokens": sum(len(m.content or "") for m in messages) + len(response_text),
            },
        }
    
    def _create_completion_stream(
        self,
        model: str,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        n: int = 1,
        stop: Optional[Union[str, List[str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Create streaming chat completion."""
        # Generate response
        response_text = self._generate_response(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
        )
        
        # Stream response character by character
        for i, char in enumerate(response_text):
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": None,
                    }
                ],
            }
            yield chunk
        
        # Final chunk
        yield {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
