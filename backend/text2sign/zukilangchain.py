from typing import Any, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from openai import OpenAI
from pydantic import Field

class CustomLM(LLM):
    api_key: str = Field(..., title="API Key", description="OpenAI API Key")
    base_url: str = Field(..., title="Base URL", description="OpenAI Base URL")
    model_name: str = Field(..., title="Model Name", description="OpenAI Model Name")

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        chat_completions = client.chat.completions.create(
            stream=False,
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return chat_completions.choices[0].message.content


### USAGE
# llm = CustomLM(api_key="zu-<ZUKIAPIKEY>", base_url="https://zukijourney.xyzbot.net/v1", model_name="gpt-4")
# llm.invoke('hi how are you doing')
