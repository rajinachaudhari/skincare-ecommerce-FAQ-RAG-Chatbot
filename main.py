from transformers import pipeline

from langchain_huggingface import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

# ✅ Import new class-based retriever
from retrieval.retriever import FaissSemanticRetriever


# ==============================
# Initialize FAISS retriever instance
# ==============================
faiss_retriever_instance = FaissSemanticRetriever()


# ==============================
# LLM Setup
# ==============================
llm_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)


# ==============================
# Chat Memory
# ==============================
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ==============================
# LangChain Wrapper Retriever
# ==============================
class FaissRetriever(BaseRetriever):
    top_k: int = 3

    def _get_relevant_documents(self, query: str):
        results_df = faiss_retriever_instance.semantic_search(
            query,
            top_k=self.top_k
        )

        docs = []
        for _, row in results_df.iterrows():
            docs.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "score": row.get("score", 0.0),
                        "domain": row.get("domain", "")
                    }
                )
            )
        return docs

    async def _aget_relevant_documents(self, query: str):
        return self._get_relevant_documents(query)


retriever = FaissRetriever(top_k=3)


# ==============================
# Prompt Template
# ==============================
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful skincare ecommerce assistant. "
        "Use ONLY the given context to answer. "
        "If the answer is not in the context, say you do not know."
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ==============================
# RAG Chain
# ==============================
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# ==============================
# Add Conversation Memory
# ==============================
chat_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)


# ==============================
# Terminal Chat Loop
# ==============================
if __name__ == "__main__":
    print("\nWelcome to Skincare FAQ RAG Chatbot")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Thank you for using the chatbot. Goodbye!")
            break

        print("Running chain...")

        answer = chat_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": "terminal"}}
        )

        print(f"Bot: {answer}\n")