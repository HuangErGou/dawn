from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class Joke(BaseModel):
    setup: str = Field(description="笑话的设定")
    punchline: str = Field(description="笑话的妙语")

print(Joke.schema())

model = ChatOpenAI(model="glm-4", openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
structured_llm = model.with_structured_output(Joke, include_raw=True)


response = structured_llm.invoke("给我讲一个关于猫的笑话")
print(response)



