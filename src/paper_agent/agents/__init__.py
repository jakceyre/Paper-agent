"""LangGraph node implementations — orchestrate tools + LLM calls.

Each agent node is a function that:
1. Takes AgentState as input
2. Calls tools and/or LLM
3. Returns a partial state dict that LangGraph merges into the full state.
"""

from paper_agent.agents.planner import plan
from paper_agent.agents.ranker import rank
from paper_agent.agents.synthesizer import synthesize
from paper_agent.agents.reviewer import review

__all__ = ["plan", "rank", "synthesize", "review"]
