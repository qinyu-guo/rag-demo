import streamlit as st
from rag import answer

st.set_page_config(page_title="RAG 问答 Demo", page_icon="🤖")
st.title("检索增强生成（RAG）问答 Demo")
st.caption("语料：RAG / 大模型基础介绍（5 段中文文档）")

q = st.text_input("输入你的问题", placeholder="例如：什么是 RAG？")

if q:
    with st.spinner("检索中..."):
        ans, hits = answer(q)

    st.subheader("💡 回答")
    st.write(ans)

    st.subheader("📚 引用来源（按相关度排序）")
    for i, (text, score) in enumerate(hits, 1):
        st.markdown(f"**{i}.** 相似度 `{score:.3f}`")
        st.write(text[:120] + ("..." if len(text) > 120 else ""))
        st.divider()