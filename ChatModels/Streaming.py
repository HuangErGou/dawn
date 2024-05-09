from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="glm-4",
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/")

for chunk in chat.stream(input="给我写一首关于月亮上金鱼的歌"):
    print(chunk.content, end="", flush=True)
