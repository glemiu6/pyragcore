#pyragcore/llm/responder.py
from langchain_ollama import OllamaLLM
from pyragcore.llm.prompt import build_prompt

class Responder:
    def __init__(self,model_name="llama3.2"):
        self.model=OllamaLLM(model=model_name)



    def _call_llm(self,prompt):
        response=self.model.generate([prompt])
        return response.generations[0][0].text

    def _stream_llm(self,prompt):
        full_response=""
        for chunk in self.model.stream(prompt):
            print(chunk,end="",flush=True)
            full_response+=chunk
        print()
        return full_response


    def answer(self,question:str,context:str,chat_history:list[dict[str,str]]|None=None,stream:bool=False):
        prompt=build_prompt(context,question,chat_history=chat_history)
        if stream:
            return self._stream_llm(prompt)
        return self._call_llm(prompt)


