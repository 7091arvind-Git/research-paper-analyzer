ANALYST_PROMPT_TEMPLATE = """You are a research paper analyst. Answer the user's question using only the provided research paper context. Do not invent information. If the answer is not available in the context, clearly say that the information is not available in the paper.

Research Paper Context:
{context}

User Question:
{question}

Answer:"""

CRITIC_PROMPT_TEMPLATE = """You are a critic agent evaluating a generated answer against the retrieved research paper context.

Evaluation criteria:
1. Is the answer relevant to the question?
2. Is every claim in the answer strictly supported by the retrieved context?
3. Are there any unsupported claims or hallucinations?

Research Paper Context:
{context}

User Question:
{question}

Generated Answer:
{answer}

Respond ONLY with a JSON object in this exact format:
{{
    "status": "SUPPORTED" or "NEEDS_REVISION",
    "feedback": "Brief explanation of why the answer is supported or what needs revision"
}}"""

REVISION_PROMPT_TEMPLATE = """You are a research paper analyst revising an answer based on critic feedback.

Research Paper Context:
{context}

User Question:
{question}

Previous Answer:
{previous_answer}

Critic Feedback:
{feedback}

Instructions:
Revise the answer to remove any unsupported claims or hallucinations. Use ONLY the provided context. If the information is not present in the context, state that clearly.

Revised Answer:"""

SUMMARY_PROMPT_TEMPLATE = """You are a research paper analyst. Based ONLY on the provided research paper context, provide a structured summary covering:
1. Research Problem & Objective
2. Methodology & Proposed Approach
3. Dataset & Experiments
4. Main Results & Key Findings
5. Limitations
6. Future Work

If any of these aspects are not mentioned in the context, clearly state that the paper does not mention them.

Research Paper Context:
{context}

Summary:"""
