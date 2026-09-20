import json
import config
from rag.embeddings import get_client
from utils.prompts import CRITIC_PROMPT_TEMPLATE
from agents.analyst_agent import format_context, generate_llm_response

def critic_answer(question, answer, chunks, client=None, api_key=None):
    if client is None:
        client = get_client(api_key)
    context = format_context(chunks)
    prompt = CRITIC_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        answer=answer
    )
    raw_text = generate_llm_response(client, prompt)
    try:
        clean_text = raw_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0].strip()
        data = json.loads(clean_text)
        status = data.get("status", "SUPPORTED").strip()
        feedback = data.get("feedback", "").strip()
        return {
            "status": status,
            "feedback": feedback
        }
    except Exception:
        status = "SUPPORTED" if "SUPPORTED" in raw_text.upper() else "NEEDS_REVISION"
        return {
            "status": status,
            "feedback": raw_text
        }
