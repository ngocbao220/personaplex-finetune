"""Response quality evaluation for post-interruption speech."""

from __future__ import annotations

import os
import re


def _clean_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 2}


def compute_response_quality(
    interruption_query: str,
    agent_response: str,
    pre_interruption_agent_speech: str = "",
    use_llm: bool = False,
    api_key: str | None = None,
) -> float:
    """Evaluates the semantic quality and relevance of the agent's response post-interruption.

    Scale: 1.0 (very poor) to 5.0 (excellent).
    """
    if not agent_response.strip():
        return 1.0

    # Optional: LLM Judge if requested and key is available
    key = api_key or os.getenv("OPENAI_API_KEY")
    if use_llm and key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            prompt = (
                "You are an expert dialogue judge. Evaluate the conversational agent's response "
                "following a user interruption on a 1-5 scale (1: terrible, 5: excellent).\n\n"
                f"Prior Agent Speech: {pre_interruption_agent_speech}\n"
                f"User Interruption: {interruption_query}\n"
                f"Agent Response Post-Interruption: {agent_response}\n\n"
                "Criteria:\n"
                "- Did the agent acknowledge or address the interruption?\n"
                "- Did it avoid merely repeating what it was saying before?\n"
                "- Is the response coherent and contextually appropriate?\n\n"
                "Output ONLY a single floating-point number between 1.0 and 5.0."
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            score_str = completion.choices[0].message.content.strip()
            score = float(re.search(r"([1-5](\.\d+)?)", score_str).group(1))
            return round(min(5.0, max(1.0, score)), 2)
        except Exception:
            pass

    # Offline heuristic fallback:
    # 1. Relevance: overlap with interruption keywords
    query_tokens = _clean_tokens(interruption_query)
    resp_tokens = _clean_tokens(agent_response)
    pre_tokens = _clean_tokens(pre_interruption_agent_speech)

    overlap = len(query_tokens.intersection(resp_tokens))
    rep_overlap = len(pre_tokens.intersection(resp_tokens)) if pre_tokens else 0

    base_score = 3.0
    if len(resp_tokens) >= 3:
        base_score += 0.5
    if overlap > 0:
        base_score += min(1.5, overlap * 0.5)
    # Penalize blindly repeating prior speech if query changed topic
    if pre_tokens and rep_overlap > len(resp_tokens) * 0.7:
        base_score -= 1.0

    return round(min(5.0, max(1.0, base_score)), 2)
