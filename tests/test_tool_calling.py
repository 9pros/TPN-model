"""
Tests for tool calling system.
"""

import pytest
from tpn_engine.tool_calling import Tool, ToolCall, ToolRegistry, ToolExecutor


class TestTool:
    """Test tool definition."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            }
        )
        
        assert tool.name == "add"
        assert tool.description == "Add two numbers"
        assert tool.function(1, 2) == 3
    
    def test_tool_to_openai_schema(self):
        """Test converting tool to OpenAI schema."""
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(
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
        
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "add"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "b" in schema["function"]["parameters"]["properties"]


class TestToolCall:
    """Test tool call."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall(
            id="call_1",
            type="function",
            function_name="add",
            arguments={"a": 1, "b": 2}
        )
        
        assert call.id == "call_1"
        assert call.function_name == "add"
        assert call.arguments == {"a": 1, "b": 2}
    
    def test_tool_call_from_dict(self):
        """Test creating tool call from dictionary."""
        data = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "add",
                "arguments": '{"a": 1, "b": 2}'
            }
        }
        
        call = ToolCall.from_dict(data)
        assert call.id == "call_1"
        assert call.function_name == "add"
        assert call.arguments == {"a": 1, "b": 2}


class TestToolRegistry:
    """Test tool registry."""
    
    def test_registry_creation(self):
        """Test creating a tool registry."""
        registry = ToolRegistry()
        assert len(registry.tools) == 0
    
    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(
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
        
        registry.register(tool)
        assert "add" in registry.tools
        assert registry.get("add") is not None
    
    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        tool = Tool(
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
        
        registry.register(tool)
        retrieved = registry.get("add")
        assert retrieved is not None
        assert retrieved.name == "add"
    
    def test_list_tools(self):
        """Test listing all tools."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        def subtract(a: int, b: int) -> int:
            return a - b
        
        registry.register(Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        registry.register(Tool(
            name="subtract",
            description="Subtract two numbers",
            function=subtract,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        tools = registry.list_tools()
        assert len(tools) == 2
        assert "add" in tools
        assert "subtract" in tools


class TestToolExecutor:
    """Test tool executor."""
    
    def test_executor_creation(self):
        """Test creating a tool executor."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        assert executor.registry is registry
    
    def test_execute_tool_call(self):
        """Test executing a tool call."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        executor = ToolExecutor(registry)
        call = ToolCall(
            id="call_1",
            type="function",
            function_name="add",
            arguments={"a": 1, "b": 2}
        )
        
        result = executor.execute(call)
        assert result == 3
    
    def test_execute_multiple_tool_calls(self):
        """Test executing multiple tool calls."""
        registry = ToolRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        def multiply(a: int, b: int) -> int:
            return a * b
        
        registry.register(Tool(
            name="add",
            description="Add two numbers",
            function=add,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        registry.register(Tool(
            name="multiply",
            description="Multiply two numbers",
            function=multiply,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        executor = ToolExecutor(registry)
        
        calls = [
            ToolCall(id="call_1", type="function", function_name="add", arguments={"a": 1, "b": 2}),
            ToolCall(id="call_2", type="function", function_name="multiply", arguments={"a": 3, "b": 4}),
        ]
        
        results = executor.execute_multiple(calls)
        assert len(results) == 2
        assert results[0] == 3
        assert results[1] == 12
    
    def test_execute_with_error(self):
        """Test executing tool call with error."""
        registry = ToolRegistry()
        
        def divide(a: int, b: int) -> float:
            return a / b
        
        registry.register(Tool(
            name="divide",
            description="Divide two numbers",
            function=divide,
            parameters={"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}
        ))
        
        executor = ToolExecutor(registry)
        call = ToolCall(
            id="call_1",
            type="function",
            function_name="divide",
            arguments={"a": 1, "b": 0}
        )
        
        result = executor.execute(call)
        assert "error" in str(result).lower()
