import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import config

load_dotenv()

embeddings = OllamaEmbeddings(model=config.embedding_model)
vector_store = PineconeVectorStore(
    pinecone_api_key=config.pinecone_api_key,
    index_name=config.pinecone_index_name,
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
        model=ChatGroq(model=config.model_name, api_key=config.groq_api_key),
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

    return {"answer": answer, "context": context_docs}


if __name__ == "__main__":
    response = run_llm("Why should I use LangChain?")
    print(response.get("answer", "No answer generated."))
