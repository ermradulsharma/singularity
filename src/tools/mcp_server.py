import json
import os
import ast
import asyncio
from typing import Dict, Any, List, Optional

def _format_jsonrpc_response(request_id: Any, result: Any = None, error: Any = None) -> Dict[str, Any]:
    response = {"jsonrpc": "2.0", "id": request_id}
    if error:
        response["error"] = error
    else:
        response["result"] = result
    return response

class MCPServer:
    """Model Context Protocol (MCP) JSON-RPC 2.0 tool server supporting dynamic tool discovery and execution."""

    def __init__(self, tools_dir: Optional[str] = None):
        """Initializes the MCP Server with dynamic tool registry from the tools directory."""
        self.tools_dir = tools_dir or os.path.dirname(os.path.abspath(__file__))
        self.tool_registry = {}
        self._discover_tools()

    def _discover_tools(self) -> None:
        if not os.path.exists(self.tools_dir):
            return
        for filename in os.listdir(self.tools_dir):
            if filename.endswith(".py") and not filename.startswith("__") and filename != "mcp_server.py":
                filepath = os.path.join(self.tools_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                            doc = ast.get_docstring(node) or "No description provided."
                            params = [a.arg for a in node.args.args if a.arg != "self"]
                            self.tool_registry[node.name] = {
                                "file": filename,
                                "module": filename[:-3],
                                "description": doc.splitlines()[0],
                                "parameters": params
                            }
                except Exception:
                    pass

    def list_tools(self) -> List[Dict[str, Any]]:
        """Lists all assimilated tools in standard MCP JSON-RPC tool schema format."""
        tools_list = []
        for name, meta in self.tool_registry.items():
            tools_list.append({
                "name": name,
                "description": meta["description"],
                "inputSchema": {
                    "type": "object",
                    "properties": {p: {"type": "string", "description": f"Parameter {p}"} for p in meta["parameters"]},
                    "required": meta["parameters"]
                }
            })
        return tools_list

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered tool by name with arguments and returns MCP content result with dynamic type coercion."""
        if name not in self.tool_registry:
            return {"isError": True, "content": [{"type": "text", "text": f"Tool '{name}' not found in MCP registry."}]}
        meta = self.tool_registry[name]
        try:
            import importlib, inspect
            mod = importlib.import_module(f"src.tools.{meta['module']}")
            func = getattr(mod, name)
            
            # Coerce argument types based on function signature inspection
            coerced_args = {}
            sig = inspect.signature(func)
            for k, v in arguments.items():
                if k in sig.parameters:
                    param_type = sig.parameters[k].annotation
                    if param_type == int and isinstance(v, str) and v.isdigit():
                        coerced_args[k] = int(v)
                    elif param_type == float and isinstance(v, str):
                        try: coerced_args[k] = float(v)
                        except ValueError: coerced_args[k] = v
                    elif param_type == bool and isinstance(v, str):
                        coerced_args[k] = v.lower() in ("true", "1", "yes")
                    else:
                        coerced_args[k] = v
                else:
                    coerced_args[k] = v

            res = func(**coerced_args)
            return {"isError": False, "content": [{"type": "text", "text": str(res)}]}
        except Exception as e:
            return {"isError": True, "content": [{"type": "text", "text": f"MCP Tool Execution Error: {e}"}]}

    def handle_jsonrpc_request(self, request_json: str) -> str:
        """Processes an incoming MCP JSON-RPC 2.0 request string and returns a JSON response."""
        try:
            req = json.loads(request_json)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            if method == "initialize":
                res = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}},
                    "serverInfo": {"name": "singularity-mcp-server", "version": "2.0.0"}
                }
                return json.dumps(_format_jsonrpc_response(req_id, result=res))
            elif method == "tools/list":
                return json.dumps(_format_jsonrpc_response(req_id, result={"tools": self.list_tools()}))
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_args = params.get("arguments", {})
                result = self.call_tool(tool_name, tool_args)
                return json.dumps(_format_jsonrpc_response(req_id, result=result))
            else:
                err = {"code": -32601, "message": f"Method '{method}' not found."}
                return json.dumps(_format_jsonrpc_response(req_id, error=err))
        except Exception as e:
            return json.dumps(_format_jsonrpc_response(None, error={"code": -32700, "message": f"Parse Error: {e}"}))

class MCPClient:
    """Model Context Protocol (MCP) client for dispatching JSON-RPC requests to MCP servers."""

    def __init__(self, server: MCPServer):
        """Initializes the MCP Client bound to a local or remote MCPServer instance."""
        self.server = server

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Sends a tools/list request to the MCP server and returns the available tools schema."""
        req = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        resp_str = self.server.handle_jsonrpc_request(req)
        resp = json.loads(resp_str)
        return resp.get("result", {}).get("tools", [])

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Dispatches a tools/call request to the MCP server and returns text content response."""
        req = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        })
        resp_str = self.server.handle_jsonrpc_request(req)
        resp = json.loads(resp_str)
        if "error" in resp:
            return f"MCP Error: {resp['error'].get('message')}"
        res = resp.get("result", {})
        content = res.get("content", [])
        if content and isinstance(content, list):
            return content[0].get("text", "")
        return str(res)
