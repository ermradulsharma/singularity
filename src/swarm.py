import multiprocessing
import time
import asyncio
from src.tools.recon_engine import UnrestrictedAgentReconEngine

async def _async_agent_loop(role: str, task_description: str, session, router):
    from src.inference import generate_text
    
    max_steps = 15
    for step in range(max_steps):
        # 1. BRAIN GENERATES TOOL CALL (OR FINAL ANSWER)
        prompt_context = session.get_formatted_history()
        llm_response = generate_text(prompt_context, variant=role)
            
        session.add_message("agent", llm_response)
        
        # 2. PARSE & EXECUTE (Concurrent Action Phase)
        tool_result = await router.parse_and_execute(llm_response)
        
        # 🚨 SANDBOX EXECUTION PIPELINE
        # If the model wrote python code in a markdown block, execute it!
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
            
            # 3. SAVE OBSERVATION (Short-Term Memory Phase)
            session.add_message("observation", str(observation)[:2000] + "...")
            
            # 4. LOOP BACK: Brain will read this observation and decide next step!
            continue
        else:
            # No tool was called, meaning the LLM has synthesized its final thought!
            result = f"[{role}] {llm_response}"
            return result
            
    return f"[{role}] Task terminated after {max_steps} steps to prevent infinite loops."

def sub_agent_task(role: str, task_description: str, return_dict: dict, lock):
    """Executes a sub-agent process with a specific persona/role using ReAct (Reasoning + Acting)"""
    from src.chat_session import SessionManager
    from src.tool_router import AsyncDynamicToolRouter
    from src.tools.memory_retriever import search_memory
    
    # Load Short-Term Memory
    session = SessionManager(session_id=role)
    
    # 🧠 LONG-TERM MEMORY INJECTION
    past_context = search_memory(task_description)
    
    # Strict ReAct System Prompt
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
    
    # Run the asynchronous agent loop
    result = asyncio.run(_async_agent_loop(role, task_description, session, router))
    
    # THREAD-SAFE STATE MUTATION
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
    
    # Run the actual LLM to evaluate the clones!
    judge_response = generate_text(evaluation_prompt)
    
    with lock:
        return_dict["critic_evaluation"] = {
            "status": "LLM_JUDGED",
            "feedback": judge_response
        }

def orchestrate_swarm(task_description: str, roles: list) -> dict:
    """
    SWARM INTELLIGENCE ORCHESTRATOR (ACTOR-CRITIC)
    Spawns clones to solve problems, then spawns a Critic to evaluate them.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    lock = manager.Lock()
    jobs = []
    
    # PHASE 1: GENERATION (Actors)
    for role in roles:
        p = multiprocessing.Process(target=sub_agent_task, args=(role, task_description, return_dict, lock))
        p.daemon = True # 🚀 FIX: Prevent zombie processes on parent crash
        jobs.append(p)
        p.start()
        
    for p in jobs:
        p.join()
        
    generation_results = dict(return_dict)
    
    # PHASE 2: CRITIC EVALUATION
    critic_job = multiprocessing.Process(target=critic_agent_task, args=(task_description, generation_results, return_dict, lock))
    critic_job.daemon = True
    critic_job.start()
    critic_job.join()
        
    return dict(return_dict)
