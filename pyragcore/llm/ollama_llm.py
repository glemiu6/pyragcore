#pyragcore/llm/ollama_llm.py
from pyragcore.interfaces.base_llm import BaseLLM
from langchain_ollama import OllamaLLM
from pyragcore.llm.prompt import build_prompt


class OllamaResponder(BaseLLM):
    def __init__(self, model_name="llama3.2:2b", base_url: str | None = None):
        kwargs = {"model": model_name}
        if base_url:
            kwargs["base_url"] = base_url
        self.model = OllamaLLM(**kwargs)

    def generate(self, prompt: str):
        response = self.model.generate([prompt])
        return response.generations[0][0].text

    def stream(self, prompt: str):
        full_response = ""
        for chunk in self.model.stream(prompt):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        return full_response

    def answer(self, question: str, context: str,
               chat_history: list[dict[str, str]] | None = None, stream: bool = False):
        prompt = build_prompt(context, question, chat_history=chat_history)
        if stream:
            return self.stream(prompt)
        return self.generate(prompt)