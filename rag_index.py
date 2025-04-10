# rag_index.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class AnomalyRAG:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.embedder = SentenceTransformer(model_name)
        self.docs = []
        self.embeddings = []

    def add_anomaly(self, text, metadata):
        self.docs.append({'text': text, 'meta': metadata})
        embedding = self.embedder.encode(text)
        self.embeddings.append(embedding)

    def retrieve(self, query, top_k=3):
        query_vec = self.embedder.encode(query)
        sims = cosine_similarity([query_vec], self.embeddings)[0]
        top_indices = sims.argsort()[-top_k:][::-1]
        return [self.docs[i] for i in top_indices]

    def format_context(self, docs):
        return "\n".join([f"Frame {d['meta']['frame']}: {d['text']}" for d in docs])
