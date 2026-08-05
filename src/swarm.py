import multiprocessing
import time
import asyncio
from src.tools.recon_engine import UnrestrictedAgentReconEngine

async def _async_agent_loop(role: str, task_description: str, session, router):
    from src.inference import generate_text
    
    max_steps = 5
    for step in range(max_steps):
        # 1. BRAIN GENERATES TOOL CALL (OR FINAL ANSWER)
        prompt_context = session.get_formatted_history()
        llm_response = generate_text(prompt_context)
            
        session.add_message("agent", llm_response)
        
        # 2. PARSE & EXECUTE (Concurrent Action Phase)
        tool_result = await router.parse_and_execute(llm_response)
        
        if tool_result["tool_called"]:
            observation = tool_result["observation"]
            
            # 3. SAVE OBSERVATION (Short-Term Memory Phase)
            session.add_message("observation", str(observation)[:2000] + "...")
            
            # 4. LOOP BACK: Brain will read this observation and decide next step!
            continue
        else:
            # No tool was called, meaning the LLM has synthesized its final thought!
            result = f"[{role}] {llm_response}"
            return result
            
    return f"[{role}] Task terminated after {max_steps} steps to prevent infinite loops."

def sub_agent_task(role: str, task_description: str, return_dict: dict):
    """Executes a sub-agent process with a specific persona/role using ReAct (Reasoning + Acting)"""
    from src.chat_session import SessionManager
    from src.tool_router import AsyncDynamicToolRouter
    
    # Load Short-Term Memory
    session = SessionManager(session_id=role)
    session.add_message("user", task_description)
    
    router = AsyncDynamicToolRouter()
    
    # Run the asynchronous agent loop
    result = asyncio.run(_async_agent_loop(role, task_description, session, router))
    return_dict[role] = result

def critic_agent_task(task_description: str, generation_results: dict, return_dict: dict):
    """
    CRITIC AGENT
    Reviews the outputs from the sub-agents and selects the preferred solution.
    """
    time.sleep(1.0) # Simulating review process
    
    # A real critic would use an LLM or logic checker. Here we simulate preference ranking.
    scores = {}
    for role, result in generation_results.items():
        # Mock scoring metric (in reality, an LLM grades it)
        scores[role] = len(result)
        
    best_role = max(scores, key=scores.get) if scores else None
    
    return_dict["critic_evaluation"] = {
        "scores": scores,
        "preferred_role": best_role,
        "feedback": f"Selected {best_role} as the best approach for the problem."
    }

def orchestrate_swarm(task_description: str, roles: list) -> dict:
    """
    SWARM INTELLIGENCE ORCHESTRATOR (ACTOR-CRITIC)
    Spawns clones to solve problems, then spawns a Critic to evaluate them.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    
    # PHASE 1: GENERATION (Actors)
    for role in roles:
        p = multiprocessing.Process(target=sub_agent_task, args=(role, task_description, return_dict))
        p.daemon = True # 🚀 FIX: Prevent zombie processes on parent crash
        jobs.append(p)
        p.start()
        
    for p in jobs:
        p.join()
        
    generation_results = dict(return_dict)
    
    # PHASE 2: CRITIC EVALUATION
    critic_job = multiprocessing.Process(target=critic_agent_task, args=(task_description, generation_results, return_dict))
    critic_job.daemon = True
    critic_job.start()
    critic_job.join()
        
    return dict(return_dict)
