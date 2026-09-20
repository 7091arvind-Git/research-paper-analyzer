import time
import config
from rag.embeddings import get_client
from utils.prompts import ANALYST_PROMPT_TEMPLATE, REVISION_PROMPT_TEMPLATE

def generate_llm_response(client, prompt):
    models = getattr(config, "FALLBACK_LLM_MODELS", [config.LLM_MODEL])
    last_error = None
    for model_name in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                last_error = e
                time.sleep(1)
    raise last_error

def format_context(chunks):
    if isinstance(chunks, str):
        return chunks
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[Page {chunk['page']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}")
    return "\n\n".join(context_parts)

def analyze_research(question, chunks, client=None, api_key=None):
    if client is None:
        client = get_client(api_key)
    context = format_context(chunks)
    prompt = ANALYST_PROMPT_TEMPLATE.format(context=context, question=question)
    return generate_llm_response(client, prompt)

def revise_answer(question, chunks, previous_answer, feedback, client=None, api_key=None):
    if client is None:
        client = get_client(api_key)
    context = format_context(chunks)
    prompt = REVISION_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        previous_answer=previous_answer,
        feedback=feedback
    )
    return generate_llm_response(client, prompt)
