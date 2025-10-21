import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Simple RAG", layout="wide")
st.title("Coffee FAQ Chatbot")

# Load exactly what you saved
index = faiss.read_index("index.faiss")
lines = np.load("lines.npy", allow_pickle=True).tolist()
model = SentenceTransformer("all-MiniLM-L6-v2")


user_question = st.text_input("Ask your question:")
if user_question:
    q_emb = model.encode([user_question])
    D, I = index.search(np.array(q_emb), k=1)
    st.write("Answer:", lines[I[0][0]])



#query = st.text_input("Enter your question:")
#top_k = st.slider("Top K", 1, 5, 3)

#if query:
 #   q = model.encode([query], convert_to_numpy=True).astype("float32")
  #  D, I = index.search(q, top_k)
   # st.subheader("Top Results:")
    #for r, (d, i) in enumerate(zip(D[0], I[0]), 1):
     #   st.write(f"{r}. {lines[i]}  (distance={d:.4f})")

