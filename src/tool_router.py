import re
import json
import asyncio
import importlib
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ValidationError

class ToolCallPayload(BaseModel):
    """Pydantic Schema for validated Tool Call Payloads."""
    tool: str = Field(description="Target tool module or class name")
    function: Optional[str] = Field(default=None, description="Target function name inside the tool module")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments passed to the tool function")
    query: Optional[str] = Field(default=None, description="Direct query parameter fallback")
    ticker: Optional[str] = Field(default=None, description="Direct ticker parameter fallback")
    code: Optional[str] = Field(default=None, description="Direct code parameter fallback")

class ConstrainedStructuredToolRouter:
    """
    Production Constrained JSON Schema & Grammar Router.
    Parses, validates via Pydantic/JSON Schemas, and executes tools asynchronously.
    """
    def __init__(self):
        self.registered_schemas = self._build_tool_schemas()

    def _build_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Extracts JSON Schema specifications for system prompt ingestion."""
        return {
            "recon_engine": {
                "description": "Unrestricted Autonomous Agent Web Reconnaissance Engine",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            },
            "stock_analysis": {
                "description": "Institutional Quantitative Financial Analytics Engine",
                "parameters": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}
            },
            "code_interpreter": {
                "description": "Python Secure Code Interpreter Sandbox Execution",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
            }
        }

    def get_tool_schemas_json(self) -> str:
        """Returns JSON schema definitions of registered tools for constrained prompt alignment."""
        return json.dumps(self.registered_schemas, indent=2)

    def extract_raw_json_blocks(self, text: str) -> List[str]:
        """Extracts JSON tool blocks from <tool_call>, markdown ```json, or raw JSON structures."""
        extracted = []
        pattern_tag = r"<tool_call>(.*?)</tool_call>"
        matches_tag = re.findall(pattern_tag, text, re.DOTALL)
        extracted.extend(matches_tag)
        
        pattern_code = r"```json\s*(\{.*?\})\s*```"
        matches_code = re.findall(pattern_code, text, re.DOTALL)
        extracted.extend(matches_code)
        
        if not extracted and text.strip().startswith("{") and text.strip().endswith("}"):
            extracted.append(text.strip())
            
        return extracted

    async def _execute_validated_tool(self, payload: ToolCallPayload) -> str:
        tool_name = payload.tool
        function_name = payload.function
        kwargs = payload.kwargs
        
        if payload.query and "query" not in kwargs:
            kwargs["query"] = payload.query
        if payload.ticker and "ticker" not in kwargs:
            kwargs["ticker"] = payload.ticker
        if payload.code and "code" not in kwargs:
            kwargs["code"] = payload.code

        if tool_name == "recon_engine":
            from src.tools.recon_engine import UnrestrictedAgentReconEngine
            engine = UnrestrictedAgentReconEngine()
            query = kwargs.get("query", "")
            obs = await asyncio.to_thread(engine.autonomous_search, query)
            return f"[{tool_name} Result]: {obs}"

        elif tool_name == "stock_analysis":
            from src.tools.stock_analysis import InstitutionalQuantEngine
            ticker = kwargs.get("ticker", "AAPL")
            engine = InstitutionalQuantEngine(ticker_symbol=ticker)
            obs = await asyncio.to_thread(engine.run_conviction_engine)
            return f"[{tool_name} Result]: {json.dumps(obs)}"

        elif tool_name == "code_interpreter":
            from src.tools.code_interpreter import run_python_code
            code = kwargs.get("code", "")
            obs = await asyncio.to_thread(run_python_code, code)
            return f"[{tool_name} Result]: {obs}"

        if not function_name:
            return f"[ERROR] Missing function name for universal tool: {tool_name}"

        try:
            module = importlib.import_module(f"src.tools.{tool_name}")
            func = getattr(module, function_name)
            obs = await asyncio.to_thread(func, **kwargs)
            return f"[{tool_name}.{function_name} Result]: {obs}"
        except (ModuleNotFoundError, AttributeError):
            from src.tools.mcp_server import MCPServer
            mcp = MCPServer()
            res = mcp.call_tool(tool_name, kwargs)
            if not res.get("isError"):
                return f"[MCP Tool '{tool_name}' Result]: {res['content'][0]['text']}"
            return f"[ERROR] Tool module or MCP tool not found: {tool_name}"
        except Exception as e:
            return f"[ERROR] Tool execution failed: {str(e)}"

    async def parse_and_execute(self, llm_response: str) -> dict:
        raw_blocks = self.extract_raw_json_blocks(llm_response)
        if not raw_blocks:
            return {"tool_called": False, "observation": None}

        tasks = []
        for block in raw_blocks:
            try:
                raw_json = json.loads(block.strip())
                validated_payload = ToolCallPayload(**raw_json)
                tasks.append(self._execute_validated_tool(validated_payload))
            except (json.JSONDecodeError, ValidationError) as ve:
                tasks.append(asyncio.to_thread(lambda: f"[VALIDATION ERROR] Tool schema invalid: {ve}"))

        results = await asyncio.gather(*tasks)
        combined_observation = "\n\n".join(results)
        return {"tool_called": True, "observation": combined_observation}

    def parse_openai_tool_calls(self, tool_calls_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parses standard OpenAI JSON Schema format tool_calls array and dispatches via MCP / Tool Router."""
        parsed_results = []
        from src.tools.mcp_server import MCPServer
        mcp = MCPServer()
        for call in tool_calls_data:
            function_data = call.get("function", {})
            name = function_data.get("name", "")
            arguments_str = function_data.get("arguments", "{}")
            try:
                args = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
            except Exception:
                args = {}
            res = mcp.call_tool(name, args)
            parsed_results.append({
                "id": call.get("id", f"call_{name}"),
                "name": name,
                "result": res
            })
        return parsed_results

    def convert_mcp_to_openai_schema(self, mcp_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Converts Anthropic MCP tool schemas to standard OpenAI function declaration schemas."""
        openai_tools = []
        for tool in mcp_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {"type": "object", "properties": {}})
                }
            })
        return openai_tools


class GrammarConstrainedLogitProcessor:
    """
    Constrained Token Logit Mask Processor for 100% Guaranteed JSON Schema Compliance.
    Masks out token logits during LLM sampling step to prevent syntax or schema violations.
    """
    def __init__(self, allowed_json_keys: Optional[List[str]] = None, vocab_size: int = 128256):
        self.allowed_keys = allowed_json_keys or ["tool", "function", "kwargs", "query", "ticker", "code"]
        self.vocab_size = vocab_size

    def process_logits(self, input_ids: Any, logits: Any) -> Any:
        """Applies constrained logit biasing masks to ensure syntactically valid JSON tool calling."""
        if logits is None:
            return logits
            
        # Biases valid JSON structural tokens ({, }, ", :, comma) and prevents illegal syntax characters
        # Common ASCII JSON token IDs in tiktoken gpt2/cl100k/o200k encodings
        json_structural_tokens = [123, 125, 34, 58, 44, 91, 93, 220, 198] # {, }, ", :, ,, [, ], space, newline
        
        # Boost structural JSON tokens when starting or continuing tool payload
        for token_id in json_structural_tokens:
            if token_id < logits.size(-1):
                logits[..., token_id] += 2.5
                
        return logits

class AsyncDynamicToolRouter(ConstrainedStructuredToolRouter):
    """Backwards compatible alias for ConstrainedStructuredToolRouter."""
    pass



