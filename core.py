import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vector_store = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
)


@tool(response_format="content_and_artifact")
def retrieve_content(query: str) -> tuple[str, List[Document]]:
    """Tool to retrieve relevant content from the vector store based on the query."""

    retrieve_content = vector_store.as_retriever().invoke(query, k=5)

    # Format the retrieved documents to include content and source information
    serialized = "\n\n".join(
        f"Content: {doc.page_content}\nSource: {doc.metadata['source'] or 'Unknown'}"
        for doc in retrieve_content
    )

    return serialized, retrieve_content


def run_llm(query: str) -> Dict[str, Any]:
    """
    Runs the LLM agent with the given query and returns the answer along with the context used.

    Args:
        query (str): The query to be answered.

    Returns:
        dict: A dictionary containing the answer and context used.
    """

    system_prompt = (
        "You are a helpful assistant that answers questions about LangChain documentation. "
        "Use the provided content to answer the question. "
        "You should use the 'retrieve_content' tool to get relevant information from the documentation. "
        "Use the tool as needed to find the information required to answer the question. "
        "If the content does not contain the answer, say you don't know. Always provide the source of the information."
    )

    agent = create_agent(
        model=ChatGroq(model=MODEL_NAME, api_key=GROQ_API_KEY),
        tools=[retrieve_content],
        system_prompt=system_prompt,
    )

    conversation = [{"role": "user", "content": query}]
    response = agent.invoke({"messages": conversation})

    messages = response.get("messages", [])
    if not messages:
        return {"answer": "No response generated.", "context": ""}

    final_message = messages[-1]
    answer = (
        final_message.content
        if hasattr(final_message, "content")
        else final_message.get("content", "")
    )

    context_docs = []
    for message in messages:
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            if message.artifact and isinstance(message.artifact, list):
                context_docs.append(message.artifact)
            context_docs.append(message.content)
        elif isinstance(message, dict) and message.get("role") == "tool":
            context_docs.append(message.get("content", ""))

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    response = run_llm("What is difference between LangChain and LangSmith?")
    print(response.get("answer", "No answer generated."))
