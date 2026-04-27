from langchain_ollama import OllamaLLM
from ..llm.prompt import build_prompt

class Responder:
    def __init__(self,model_name="llama3.2"):
        self.model=OllamaLLM(model=model_name)



    def _call_llm(self,prompt):
        response=self.model.generate([prompt])
        return response.generations[0][0].text


    def answer(self,question:str,context:str,chat_history:list[dict[str,str]]|None=None):
        prompt=build_prompt(context,question,chat_history=chat_history)
        return self._call_llm(prompt)


