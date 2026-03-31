This project implements a Retrieval‑Augmented Generation (RAG) chatbot using LangChain, FAISS, Hugging Face embeddings, and Gemini (or other LLMs).
1.Overview
It includes an evaluation harness to measure performance across multiple dimensions:
- Retrieval accuracy
- Response quality
- Hallucination rate
- Latency
- User experience

2.Architecture
Pipeline Flow:
- Retriever (FAISS + HuggingFaceEmbeddings)
- Converts user query into embeddings.
- Fetches top‑k relevant document chunks.
  
3.PromptTemplate (LangChain):
- Formats retrieved context, conversation history, and question into a structured prompt.
- LLM (Gemini / Hugging Fac)
- Generates an answer based on the prompt.
  
4.Valuation Harness:
- Measures retrieval accuracy - Precision = relevant retrieved / total retrieved.
- Measures Response quality - Recall = relevant retrieved / total gold chunks.
- Time taken to respond (latency rate)
- Hallucination rate is determined by the cosine similarity of the retrieved response with the user expected response.
5.Deployment
- Deployed in Streamlit Dashboard as the user interactive dashboard. 
