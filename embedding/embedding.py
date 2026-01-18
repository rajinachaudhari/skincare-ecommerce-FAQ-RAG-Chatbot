
###Note: embedding.npy file is saved as backup of the embeddings used to build FAISS index###

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import yaml
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


###loading config.yaml file###
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


###Loading chunked DataFrame directly###
chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

###selecting models directly###
model = SentenceTransformer(config["embedding"]["model_name"], device=config["embedding"]["device"])


embeddings = model.encode(
    chunks_df["text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)
chunks_df["embeddings"] = list(embeddings)
chunks_df.head()

###save values of embedding before passing to FAISS index i.e vector database###
np.save("data/processed/embeddings.npy", embeddings)


print("Embeddings shape:", embeddings.shape)
