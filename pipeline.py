from rag.embeddings import get_client
from agents.retriever_agent import run_retriever_agent
from agents.analyst_agent import analyze_research, revise_answer
from agents.critic_agent import critic_answer

def run_agentic_rag(question, vector_store, client=None, api_key=None):
    if client is None:
        client = get_client(api_key)
    
    chunks = run_retriever_agent(question, vector_store, client=client, api_key=api_key)
    
    if not chunks:
        return {
            "answer": "No relevant content found in the paper.",
            "sources": [],
            "critic_status": "NOT_EVALUATED",
            "critic_feedback": "No documents retrieved.",
            "revised": False,
            "initial_answer": ""
        }
    
    initial_answer = analyze_research(question, chunks, client=client, api_key=api_key)
    
    critic_result = critic_answer(question, initial_answer, chunks, client=client, api_key=api_key)
    status = critic_result.get("status", "SUPPORTED")
    feedback = critic_result.get("feedback", "")
    
    final_answer = initial_answer
    revised = False
    
    if status == "NEEDS_REVISION":
        final_answer = revise_answer(question, chunks, initial_answer, feedback, client=client, api_key=api_key)
        revised = True
        
    return {
        "answer": final_answer,
        "sources": chunks,
        "critic_status": status,
        "critic_feedback": feedback,
        "revised": revised,
        "initial_answer": initial_answer
    }

def summarize_paper(vector_store, client=None, api_key=None):
    summary_question = "Provide a comprehensive summary of the research paper covering the research problem, methodology, dataset, main results, limitations, and future work."
    return run_agentic_rag(summary_question, vector_store, client=client, api_key=api_key)
