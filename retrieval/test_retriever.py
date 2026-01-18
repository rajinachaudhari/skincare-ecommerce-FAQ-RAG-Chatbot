###Note: testing wheather FAISS cosine similarity retriever worked ###
from retriever import semantic_search_faiss

query = "Can you recommend a night cream for sensitive skin?"
results_df = semantic_search_faiss(query, top_k=3)

for i, row in results_df.iterrows():
    print(f"Domain : {row['domain']}")
    print(f"Score  : {row['score']:.4f}")
    print("Text   :")
    print(row["text"])   
