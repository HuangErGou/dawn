from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


tools = [add, multiply]

chat = ChatOpenAI(
    model="glm-4",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/")
chat_with_tools = chat.bind_tools(tools)

query = "5加上2，然后再乘以3等于多少？"

messages = [HumanMessage(query)]
ai_message = chat_with_tools.invoke(messages)

while len(ai_message.tool_calls) > 0:
    print(ai_message)
    messages.append(ai_message)

    for tool_call in ai_message.tool_calls:
        selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    print("================================\n")
    print(messages)
    print("================================\n")

    ai_message = chat_with_tools.invoke(messages)

print(ai_message)
