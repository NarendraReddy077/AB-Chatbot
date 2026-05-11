from langchain_core.prompts import PromptTemplate

QA_PROMPT_TEMPLATE = """Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say "I could not find this information in the provided documentation." Do not try to make up an answer.

Context:
{context}

Question:
{question}

Answer:"""

qa_prompt = PromptTemplate(
    template=QA_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)
