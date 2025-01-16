from langchain.callbacks.base import BaseCallbackHandler

class AgentCallbackHandler(BaseCallbackHandler):

    def on_llm_start(
            self, serialized:Dict[str,Any] , prompts: List[str], **kwargs: Any
    ) -> Any:

        print(f"**** Prompt to LLM was:**** \n{prompts[0]}")
        print("***********")

    def on_llm_end(self,response:LLMResult,**kwargs:Any) -> Any:

        print(f"*****LLM Response:*** \n{reponse.generations[0][0].text}")
        print("**************")
