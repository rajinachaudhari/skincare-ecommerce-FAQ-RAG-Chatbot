
# ###Note: embedding.npy file is saved as backup of the embeddings used to build FAISS index###

# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# import yaml
# from sentence_transformers import SentenceTransformer
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd


# ###loading config.yaml file###
# with open("config/config.yaml", "r", encoding="utf-8") as f:
#     config = yaml.safe_load(f)


# ###Loading chunked DataFrame directly###
# chunks_df = pd.read_csv(config["paths"]["chunks_csv"])

# ###selecting models directly###
# model = SentenceTransformer(config["embedding"]["model_name"], device=config["embedding"]["device"])


# embeddings = model.encode(
#     chunks_df["text"].tolist(),
#     convert_to_numpy=True,
#     show_progress_bar=True
# )
# chunks_df["embeddings"] = list(embeddings)
# chunks_df.head()

# ###save values of embedding before passing to FAISS index i.e vector database###
# np.save("data/processed/embeddings.npy", embeddings)


# print("Embeddings shape:", embeddings.shape)


import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yaml
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_config():
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_embeddings():
    """
    Generate normalized embeddings for chunked documents.
    This is an OFFLINE step.
    """

    config = load_config()

    chunks_path = config["paths"]["chunks_csv"]
    embeddings_path = config["paths"]["embeddings_file"]

    # Load chunked text
    chunks_df = pd.read_csv(chunks_path)

    # Load embedding model
    model = SentenceTransformer(
        config["embedding"]["model_name"],
        device=config["embedding"]["device"]
    )

    # Generate NORMALIZED embeddings (important for cosine similarity)
    embeddings = model.encode(
        chunks_df["text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,   #  IMPORTANT FIX
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    # Save embeddings
    np.save(embeddings_path, embeddings)

    print("Embeddings generated successfully.")
    print(f"Shape: {embeddings.shape}")
    print(f"Saved to: {embeddings_path}")


if __name__ == "__main__":
    generate_embeddings()