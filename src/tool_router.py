import re
import json
import asyncio
import importlib

class AsyncDynamicToolRouter:
    """
    Parses LLM outputs for Tool Calling syntax.
    Finds ALL <tool_call> blocks and executes them concurrently using asyncio.to_thread.
    Supports Universal Dynamic module loading for 31+ tools.
    """
    def __init__(self):
        pass

    async def _execute_single_tool(self, tool_json_str: str) -> str:
        try:
            tool_data = json.loads(tool_json_str.strip())
            tool_name = tool_data.get("tool")
            function_name = tool_data.get("function")
            kwargs = tool_data.get("kwargs", {})
            
            if not tool_name:
                return "[ERROR] Missing 'tool' in tool call."
                
            if tool_name in ["recon_engine", "stock_analysis", "code_interpreter"]:
                if tool_name == "recon_engine":
                    from src.tools.recon_engine import UnrestrictedAgentReconEngine
                    engine = UnrestrictedAgentReconEngine()
                    query = tool_data.get("query", kwargs.get("query", ""))
                    obs = await asyncio.to_thread(engine.autonomous_search, query)
                    return f"[{tool_name} Result]: {obs}"
                elif tool_name == "stock_analysis":
                    from src.tools.stock_analysis import InstitutionalQuantEngine
                    ticker = tool_data.get("ticker", kwargs.get("ticker", "AAPL"))
                    engine = InstitutionalQuantEngine(ticker_symbol=ticker)
                    obs = await asyncio.to_thread(engine.run_conviction_engine)
                    return f"[{tool_name} Result]: {json.dumps(obs)}"
                elif tool_name == "code_interpreter":
                    from src.tools.code_interpreter import run_python_code
                    code = tool_data.get("code", kwargs.get("code", ""))
                    obs = await asyncio.to_thread(run_python_code, code)
                    return f"[{tool_name} Result]: {obs}"
                    
            if not function_name:
                return f"[ERROR] Missing 'function' name for universal tool {tool_name}."
                
            try:
                module = importlib.import_module(f"src.tools.{tool_name}")
                func = getattr(module, function_name)
            except ModuleNotFoundError:
                return f"[ERROR] Tool module not found: {tool_name}"
            except AttributeError:
                return f"[ERROR] Function '{function_name}' not found in tool '{tool_name}'"
                
            observation = await asyncio.to_thread(func, **kwargs)
            return f"[{tool_name}.{function_name} Result]: {observation}"
            
        except json.JSONDecodeError:
            return "[ERROR] Invalid JSON in tool call."
        except Exception as e:
            return f"[ERROR] Tool execution failed: {str(e)}"

    async def parse_and_execute(self, llm_response: str) -> dict:
        """
        Looks for tool call syntax in the LLM response.
        Executes all matched tools concurrently.
        """
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, llm_response, re.DOTALL)
        
        if not matches:
            return {"tool_called": False, "observation": None}
            
        tasks = [self._execute_single_tool(m) for m in matches]
        
        results = await asyncio.gather(*tasks)
        
        combined_observation = "\n\n".join(results)
        
        return {"tool_called": True, "observation": combined_observation}

