import os
from typing import List

from langchain_core.embeddings import Embeddings
from zhipuai import ZhipuAI
from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate


class ZhiPuEmbeddings(Embeddings):
    def __init__(self):
        Embeddings.__init__(self)
        api_key = os.getenv("OPENAI_API_KEY")
        self._client = ZhipuAI(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        rtn = []
        for text in texts:
            response = self._client.embeddings.create(model="embedding-2", input=text)
            rtn.append(response.data[0].embedding)

        print("input str is %s, size %d" % (texts, len(rtn)))
        return rtn

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(model="embedding-2", input=text)
        return response.data[0].embedding


example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"input": "高兴", "output": "悲伤"},
    {"input": "高", "output": "矮"},
    {"input": "精力充沛", "output": "昏昏欲睡的"},
    {"input": "阳光明媚", "output": "阴沉"},
    {"input": "慌张", "output": "冷静的"},
]

# MMR（最大边际相关性）选择法
example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    ZhiPuEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # This is the number of examples to produce.
    k=2,
)

mmr_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="给出每个输入的反义词",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)

# Input is a feeling, so should select the happy/sad example as the first one
print(mmr_prompt.format(adjective="焦虑的"))

# 该对象根据与输入的相似度选择例子。它通过找到与输入具有最大余弦相似度的嵌入的例子
# Let's compare this to what we would just get if we went solely off of similarity,
# by using SemanticSimilarityExampleSelector instead of MaxMarginalRelevanceExampleSelector.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    # The list of examples available to select from.
    examples,
    # The embedding class used to produce embeddings which are used to measure semantic similarity.
    ZhiPuEmbeddings(),
    # The VectorStore class that is used to store the embeddings and do a similarity search over.
    FAISS,
    # The number of examples to produce.
    k=2,
)

similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Input: {adjective}\nOutput:",
    input_variables=["adjective"],
)
print(similar_prompt.format(adjective="焦虑的"))
