from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.outputs import ChatGeneration
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAI


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
    def execute(self) -> int:
        return self.a + self.b


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

    def execute(self) -> int:
        return self.a * self.b


tools = [Add, Multiply]

chat = ChatOpenAI(
    model="glm-4",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
chat_with_tools = chat.bind_tools(tools)
parser = PydanticToolsParser(tools=[Multiply, Add], return_id=True, first_tool_only=True)


query = "What is 3 * 12? Also, what is 11 + 49?"

messages = [HumanMessage(query)]

response = chat_with_tools.invoke(messages)
print(response.tool_calls)
