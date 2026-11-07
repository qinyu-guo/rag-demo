from sentence_transformers import SentenceTransformer
from openai import OpenAI
import faiss
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

DOCS = [
    "检索增强生成（Retrieval-Augmented Generation，简称 RAG）是一种将信息检索与大语言模型生成相结合的技术。它的核心思想是：在让大模型生成答案之前，先从外部知识库中检索出与问题相关的文档片段，再把这些片段作为参考上下文输入给大模型，让模型基于这些资料生成答案。这样可以弥补大模型训练数据过时、缺乏私有知识以及容易产生幻觉等问题。",

    "一个典型的 RAG 系统通常包含三个主要步骤。第一步是索引：将文档切分成小块，并用嵌入模型把每个文本块转换成向量，存入向量数据库。第二步是检索：将用户的问题也转换成向量，在向量数据库中找到最相似的若干个文本块。第三步是生成：把检索到的文本块与用户问题一起拼成提示词，交给大语言模型生成最终答案。",


    "嵌入（Embedding）是将文字转换成向量表示的过程。语义相近的文本，其向量在空间中的距离也更近。向量数据库通过计算向量之间的相似度（例如余弦相似度）来快速找到与查询最相关的文本。常用的向量检索方法包括暴力搜索和近似最近邻（ANN）算法，例如FAISS、HNSW 等，可以在海量向量中快速定位最相似的条目。",

    "大语言模型（Large Language Model，简称 LLM）是基于 Transformer 架构、在海量文本上训练得到的神经网络模型，例如GPT、DeepSeek、文心一言等。它能够理解并生成自然语言，完成问答、翻译、摘要、代码生成等任务。大语言模型通过学习海量语料中的模式来预测下一个词，从而具备了一定的语言理解和生成能力。",

    "大语言模型虽然能力强大，但存在训练数据时效性差、缺乏特定领域知识、容易产生幻觉（编造事实）等问题。RAG通过引入外部知识检索，让模型在回答时能够参考最新、最相关的资料，从而显著减少幻觉、提高答案的可追溯性。与微调（Fine-tuning）不同，RAG 不需要重新训练模型参数，知识更新只需更新外部知识库即可，成本更低、更灵活。"
]
     

def chunk_text(docs, size=200, overlap=40):
      chunks = []
      for d in docs:
          for i in range(0, len(d), size - overlap):
              chunks.append(d[i:i+size])
      return chunks

CHUNKS = DOCS
print("一共切了", len(CHUNKS), "块")
print("第一块是：", CHUNKS[0])

 

embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
vecs = embedder.encode(CHUNKS, normalize_embeddings=True).astype("float32")
print("向量形状：", vecs.shape)   # 应该是 (块数, 512)

index = faiss.IndexFlatIP(vecs.shape[1])
index.add(vecs)
print("索引里存了多少条：", index.ntotal)

def retrieve(query, k=3):
      q = embedder.encode([query], normalize_embeddings=True).astype("float32")
      scores, ids = index.search(q, k)
      return [(CHUNKS[i], float(scores[0][j])) for j, i in enumerate(ids[0])]

# 手动测一下：问个和语料相关的问题
for text, score in retrieve(" RAG 系统包含哪三个步骤？"):
    print(f"{score:.3f} -> {text[:60]}...")



LLM = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def answer(query):
    hits = retrieve(query)
    context = "\n\n".join(f"[{i+1}] {t}" for i, (t, _) in enumerate(hits))
    prompt = f"请仅根据以下资料回答，资料里没有就说不知道。\n\n资料：\n{context}\n\n问题：{query}\n回答："
    resp = LLM.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content, hits

if __name__ == "__main__":
    q = input("问题：")
    ans, hits = answer(q)
    print("\n回答：", ans)
    print("\n引用的来源：")
    for t, s in hits:
        print(f"  [{s:.3f}] {t[:60]}...")