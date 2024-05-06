from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

chat = ChatOpenAI(
    model="glm-4",
    # openai_api_key=api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/")

messages = [
    SystemMessage(content="你是一个得力的助手"),
    HumanMessage(content="模型正则化的目的是什么？, 请用10个字回答"),
]

response = chat.invoke(messages)
print(response.content)

print("\n====================================\n")

for chunk in chat.stream(messages):
    print(chunk.content, end="", flush=True)

print("\n====================================\n")

