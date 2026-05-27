from typing import Any, List

import streamlit as st

from core import run_llm


def format_source(context_docs: List[Any]) -> List[str]:
    """
    Formats the source of the context documents into a list of strings.

    Args:
        context_docs (List[Any]): A list of context documents.

    Returns:
        List[str]: A list of source strings.
    """
    return [
        str(meta["source"] or "Unknown")
        for doc in context_docs or []
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]


st.set_page_config(
    page_title="LangChain Documentation Helper",
    page_icon=":robot_face:",
    layout="centered",
)

st.title("LangChain Documentation Helper")
st.write("Enter a question to get an answer from the LangChain documentation.")

with st.sidebar:
    st.subheader("Sessions")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about LangChain documentation.",
            "source": [],
        }
    ]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["source"]:
            with st.expander("Source"):
                for source in message.get("source", []):
                    st.markdown(f"- {source}")

prompt = st.chat_input("Ask a question about LangChain documentation")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "source": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = run_llm(prompt)

                answer = (
                    str(response.get("answer", "")).strip() or "No answer generated."
                )
                sources = (
                    format_source(response.get("context", [""])[0])
                    if len(response.get("context", []))
                    else []
                )

                st.markdown(answer)

                if sources:
                    with st.expander("Source"):
                        for source in sources:
                            st.markdown(f"- {source}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "source": sources,
                    }
                )
        except Exception as e:
            st.error(e)
            st.exception(e)
