# pyragcore/llm/prompt.py

def build_prompt(context: str, question: str, chat_history: list[dict[str, str]] | None = None) -> str:
    """
    Build a structured prompt for the LLM.

    Args:
        context (str): The relevant documents or text to answer from.
        question (str): The user’s question.
        chat_history (list[dict], optional): Previous messages for conversational context. 
            Each dict should have {"role": "user"|"assistant", "message": str}. Defaults to None.

    Returns:
        str: The formatted prompt ready to send to the model.
    """

    # Start with instructions
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
        "If the answer is not in the context, say 'I don’t know.'\n"
        "Cite information only from the context.\n\n"
    )

    # Include previous chat if available
    if chat_history:
        prompt += "Previous conversation:\n"
        for msg in chat_history:
            role = msg.get("role", "user")
            message = msg.get("message", "")
            prompt += f"{role.capitalize()}: {message}\n"
        prompt += "\n"

    # Add context and question
    prompt += f"Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"

    return prompt