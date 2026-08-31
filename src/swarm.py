import multiprocessing
import time
import asyncio
from src.tools.recon_engine import UnrestrictedAgentReconEngine

async def _async_agent_loop(role: str, task_description: str, session, router):
    from src.inference import generate_text
    
    max_steps = 15
    for step in range(max_steps):
        prompt_context = session.get_formatted_history()
        llm_response = generate_text(prompt_context, variant=role)
            
        session.add_message("agent", llm_response)
        
        tool_result = await router.parse_and_execute(llm_response)
        
        sandbox_output = ""
        if "```python" in llm_response:
            import re
            from src.sandbox import SecureSandbox
            code_blocks = re.findall(r'```python\n(.*?)\n```', llm_response, re.DOTALL)
            if code_blocks:
                sandbox = SecureSandbox(use_docker=True)
                try:
                    out = sandbox.execute(code_blocks[0])
                    sandbox_output = "\n[SANDBOX EXECUTION RESULT]\n" + out
                except Exception as e:
                    sandbox_output = f"\n[SANDBOX ERROR]\n{e}\nAnalyze the error, correct your code, and try again."
        
        if tool_result["tool_called"] or sandbox_output:
            observation = tool_result.get("observation", "") + sandbox_output
            
            session.add_message("observation", str(observation)[:2000] + "...")
            
            continue
        else:
            result = f"[{role}] {llm_response}"
            return result
            
    return f"[{role}] Task terminated after {max_steps} steps to prevent infinite loops."

def sub_agent_task(role: str, task_description: str, return_dict: dict, lock):
    """Executes a sub-agent process with a specific persona/role using ReAct (Reasoning + Acting)"""
    from src.chat_session import SessionManager
    from src.tool_router import AsyncDynamicToolRouter
    from src.tools.memory_retriever import search_memory
    
    session = SessionManager(session_id=role)
    
    past_context = search_memory(task_description)
    
    react_system_prompt = f"""You are a specialized Swarm Agent with the role: {role}.
You operate in a strict ReAct (Reason + Act) loop. To solve the problem, you MUST follow this EXACT format:
Thought: Detail your reasoning step-by-step. What do you need to calculate or verify?
Code: Write python code inside a ```python block to calculate your thought. The code will execute in a secure sandbox.
Observation: (Wait for the system to provide the output of your code).
... (Repeat until you solve the problem)
Final Answer: Provide the mathematically/logically proven solution.

NEVER assume a final answer without verifying it via Code first. If you get a [SANDBOX ERROR], analyze it, fix your code, and try again."""
    session.add_message("system", react_system_prompt)
    
    enhanced_task = f"Past Long-Term Knowledge:\n{past_context}\n\nCurrent Task:\n{task_description}"
    
    session.add_message("user", enhanced_task)
    
    router = AsyncDynamicToolRouter()
    
    result = asyncio.run(_async_agent_loop(role, task_description, session, router))
    
    with lock:
        return_dict[role] = result

def critic_agent_task(task_description: str, generation_results: dict, return_dict: dict, lock):
    """
    CRITIC AGENT (LLM-AS-A-JUDGE)
    Uses the actual LLM to deeply evaluate the logic of the sub-agents and pick a winner.
    """
    from src.inference import generate_text
    
    evaluation_prompt = f"Task: {task_description}\n\nEvaluate the following agent responses and select the most mathematically/logically sound approach.\n"
    for role, result in generation_results.items():
        evaluation_prompt += f"\n--- {role} ---\n{result}\n"
        
    evaluation_prompt += "\nOutput your final reasoning and state which role won."
    
    judge_response = generate_text(evaluation_prompt)
    
    with lock:
        return_dict["critic_evaluation"] = {
            "status": "LLM_JUDGED",
            "feedback": judge_response
        }

def orchestrate_swarm(task_description: str, roles: list) -> dict:
    """
    SWARM INTELLIGENCE ORCHESTRATOR (ACTOR-CRITIC + TREE-OF-THOUGHT CONSENSUS)
    Executes in-process thread pool sub-agents, scores trajectories with PRM, and evaluates them with a Critic.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from src.prm import StepProcessRewardModel
    prm = StepProcessRewardModel()
    
    return_dict = {}
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max(1, len(roles))) as executor:
        futures = [
            executor.submit(sub_agent_task, role, task_description, return_dict, lock)
            for role in roles
        ]
        for f in futures:
            try:
                f.result()
            except Exception:
                pass
        
    generation_results = dict(return_dict)
    
    tot_scores = {}
    for role, text in generation_results.items():
        if isinstance(text, str):
            steps = text.split("\n")
            scores = prm.score_reasoning_steps(steps)
            tot_scores[role] = sum(scores) / max(1, len(scores))
            
    generation_results["tot_trajectory_scores"] = tot_scores
    
    try:
        critic_agent_task(task_description, generation_results, return_dict, lock)
    except Exception:
        pass
        
    return dict(return_dict)


