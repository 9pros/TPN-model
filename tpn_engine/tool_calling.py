"""
Tool calling system for TPN.

Provides OpenAI-compatible tool calling support with:
- Tool definition and registration
- Tool call execution
- OpenAI chat completions format
"""

import json
import traceback
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field


@dataclass
class Tool:
    """A tool that can be called by the model."""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
    
    def execute(self, arguments: Dict[str, Any]) -> Any:
        """Execute the tool with given arguments."""
        return self.function(**arguments)


@dataclass
class ToolCall:
    """A tool call request."""
    id: str
    type: str
    function_name: str
    arguments: Dict[str, Any]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create ToolCall from dictionary."""
        func = data.get("function", {})
        args_str = func.get("arguments", "{}")
        
        # Parse arguments (may be string or dict)
        if isinstance(args_str, str):
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = args_str
        
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "function"),
            function_name=func.get("name", ""),
            arguments=arguments,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function_name,
                "arguments": json.dumps(self.arguments),
            }
        }


class ToolRegistry:
    """Registry for available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())
    
    def get_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI schemas for all tools."""
        return [tool.to_openai_schema() for tool in self.tools.values()]
    
    def to_openai_tools(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format."""
        return self.get_schemas()


class ToolExecutor:
    """Executes tool calls."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    def execute(self, call: ToolCall) -> Any:
        """Execute a single tool call."""
        tool = self.registry.get(call.function_name)
        if tool is None:
            return {"error": f"Tool '{call.function_name}' not found"}
        
        try:
            result = tool.execute(call.arguments)
            return result
        except Exception as e:
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    def execute_multiple(self, calls: List[ToolCall]) -> List[Any]:
        """Execute multiple tool calls."""
        results = []
        for call in calls:
            result = self.execute(call)
            results.append(result)
        return results
    
    def execute_from_dict(self, data: Dict[str, Any]) -> Any:
        """Execute tool call from dictionary."""
        call = ToolCall.from_dict(data)
        return self.execute(call)
