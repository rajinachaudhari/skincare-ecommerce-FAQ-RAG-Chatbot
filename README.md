# Skincare Ecommerce FAQ RAG Chatbot 

A beginner-friendly Retrieval-Augmented Generation (RAG) chatbot designed to answer skincare and ecommerce-related questions using local, open-source language models.
This project does not require API keys and runs fully on your local machine.
---

##  Table of Contents

- [What is This Project?](#what-is-this-project)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Configure the Project](#configure-the-project)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Project Components Explained](#project-components-explained)
- [Contributing](#contributing)
- [License](#license)

---

##  What is This Project?

This is a Retrieval-Augmented Generation (RAG) chatbot built for skincare ecommerce FAQ-style customer support.
Instead of relying on cloud-based APIs, the chatbot works by:
Retrieving relevant skincare Q&A content using semantic search (FAISS)
Augmenting the retrieved content into a structured prompt
Generating answers using a local instruction-tuned language model

Why RAG?
Improves answer accuracy by grounding responses in real data
Prevents hallucination by restricting answers to retrieved context
Makes the system explainable and modular

**Key Benefits:**
-  No API costs or external service dependencies
-  Privacy-first: Everything runs locally on your machine
-  Fast retrieval using FAISS vector database
-  Beginner-friendly: Well-documented and modular
-  Customizable: Easy to adapt for your own Q&A data

---

##  Features

- Semantic Search using vector embeddings
- Conversation Memory (session-based, in-memory)
- Local Language Model (FLAN-T5)
- FAISS Vector Store for fast similarity search
- Config-driven pipeline using config.yaml
- LangChain-based RAG architecture

⚠️ Note: This project is intended for learning and demonstration purposes.

---

## 💻 System Requirements

Before getting started, ensure your system meets these requirements:

### Minimum Requirements:
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: `5 GB free space
- **Processor**: Any modern CPU (Works for low end PCs)
- **OS**: Windows, macOS, or Linux
- **Python**: Version 3.10 or higher

### Optional (Not Required):
- GPU (can speed up inference, but not required)
---

##  Installation & Setup

### Step 1: Clone the Repository

First, clone this repository to your local machine using Git:

```bash
git clone https://github.com/yourusername/skincare-ecommerce-FAQ-RAG-Chatbot.git
```

Navigate into the project folder:

```bash
cd skincare-ecommerce-FAQ-RAG-Chatbot
```

**Alternative (if you don't have Git):**
- Download the repository as a ZIP file from GitHub
- Extract it to your desired location
- Open a terminal/command prompt and navigate to the folder

### Step 2: Create a Virtual Environment (Recommended)

A virtual environment keeps your project dependencies isolated from your system Python.

**Using venv (Built-in):**
```bash
# On Windows:
python -m venv venv
venv\Scripts\activate

# On macOS/Linux:
python -m venv venv
source venv/bin/activate
```

**Using Conda (Alternative):**
```bash
conda create -n skincare-chatbot python=3.9
conda activate skincare-chatbot
```

### Step 3: Install Dependencies

All required packages are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `numpy` & `pandas`: Data processing
- `sentence-transformers`: Embeddings and semantic search
- `faiss-cpu`: Vector database for fast similarity search
- `langchain` & `langchain-huggingface`: LLM orchestration
- `transformers`: Pre-trained language models



### Step 4: Download Pre-trained Models

The first time you run the project, models are automatically downloaded:
- `paraphrase-MiniLM-L3-v2`: For embeddings (~50 MB)
- `google/flan-t5-base`: For text generation (~990 MB)

These are cached locally, so subsequent runs are instant.

### Step 5: Configure the Project

The project uses `config/config.yaml` for all settings. You can customize:

```yaml
paths:
  input_csv: data/qa_dataset.csv              # Input CSV containing Q&A data
  output_documents: data/processed/documents.txt  # Combined text documents generated from CSV
  processed_dir: data/processed               # Directory for all processed outputs

  input_documents: data/processed/documents.txt   # Input file for chunking step
  output_chunks: data/processed/chunks.csv        # Output CSV containing text chunks

  chunks_csv: data/processed/chunks.csv       # Chunked text used for embedding
  embeddings_file: data/processed/embeddings.npy # Saved embeddings (optional backup)

  vector_index: vectordb/faiss.index           # FAISS vector index file

processing:
  question_column: question                   # Question column name in CSV
  answer_column: answer                       # Answer column name in CSV

chunking:
  chunk_size: 80                              # Number of words per chunk
  overlap: 30                                 # Overlap between consecutive chunks
  domain: customer_support                   # Domain tag for the dataset
  encoding: utf-8                             # Text encoding for file processing

embedding:
  model_name: paraphrase-MiniLM-L3-v2         # Sentence Transformer embedding model
  device: cpu                                 # Use 'cuda' if GPU is available
  batch_size: 32                              # Number of chunks processed per batch

vectordb:
  index_type: flat_l2                         # FAISS index type (Flat L2 distance)
---

##  Project Structure

skincare-ecommerce-FAQ-RAG-Chatbot/
│
├── main.py                          # Main chatbot application
├── requirements.txt                 # Python dependencies
├── LICENSE                          # Project license
├── README.md                        # This file
│
├── config/
│   └── config.yaml                 # Configuration settings
│
├── data/
│   ├── qa_dataset.csv             # Your Q&A data (input)
│   └── processed/
│       ├── documents.txt           # Processed documents
│       ├── chunks.csv              # Text chunks for indexing
│       └── embeddings.npy          # Vector embeddings (backup)
│
├── embedding/
│   ├── embedding.py               # Generate embeddings from chunks
│   ├── chunking.py                # Split documents into chunks
│   └── csv-to-documents.py        # Convert CSV to text documents
│
├── retrieval/
│   ├── retriever.py               # Semantic search using FAISS
│   └── test_retriever.py          # Tests for retriever
│
└── vectordb/
    ├── faiss.index                # FAISS vector database (binary)
    └── vector_store.py            # Create and manage FAISS index

##  How It Works

Here's the flow of data through the chatbot:

---

##  Data Preparation Phase (One-time setup)

```
CSV Dataset → Documents → Chunks → Embeddings → FAISS Index
```

- **CSV to Documents**: Your Q&A pairs are converted to text
- **Chunking**: Long texts are split into smaller pieces (configurable)
- **Embeddings**: Each chunk is converted to a vector using `paraphrase-MiniLM-L3-v2`
- **FAISS Index**: Vectors are stored in a searchable index

---

##  Query & Response Phase (During chatbot runtime)

```
User Question → Search FAISS → Retrieve Top-3 → Prompt Augmentation → FLAN-T5 Answer Generation
```

- **User Input**: You ask a question about skincare
- **Semantic Search**: FAISS finds 3 most similar chunks (by cosine similarity)
- **Context + LLM**: Results are passed to `google/flan-t5-base` LLM
- **Prompt Augmentation**: Retrieved chunks are combined with the question in a prompt.
- **Answer Generation**: FLAN-T5 generates the final answer using the provided context.

---

##  Conversation Memory (Sessions)

- Each chat session is stored separately
- Memory is in-memory (lost when app stops)
- Easy to extend with database persistence

---

##  Usage

### Running the Chatbot

```bash
python main.py
```

### Example Interaction

```
User: Can I use retinol daily?

Bot: Retinol should be introduced gradually and used 2–3 times per week initially.
```


##  Project Components Explained

### **1. Embedding Module** (`embedding/`)

**Purpose**: Convert text into numerical vectors that capture semantic meaning

**Files**:
- `csv-to-documents.py`: Converts your CSV Q&A data into plain text
- `chunking.py`: Splits long documents into smaller, overlapping pieces
- `embedding.py`: Generates vector embeddings using `SentenceTransformer`

**Why**: Small chunks + embeddings = efficient similarity search

### **2. Vector Database** (`vectordb/`)

**Purpose**: Store and efficiently search embeddings

**Technology**: FAISS (Facebook AI Similarity Search)
- Creates an index for O-1 similarity search
- Uses Inner Product (equivalent to cosine similarity for normalized vectors)

**File**: `faiss.index` (binary file, pre-built for you)

### **3. Retrieval Module** (`retrieval/`)

**Purpose**: Find relevant Q&A chunks for a user query

**Process**:
1. Encode user query to embedding
2. Search FAISS index for top-k similar vectors
3. Return matching chunks with similarity scores
4. Include metadata (domain, source, score)

**Key Function**: `semantic_search_faiss(query, top_k=3)`

### **4. Main Application** (`main.py`)

**Purpose**: Orchestrate the entire RAG pipeline

**Components**:
- **LLM Pipeline**: Google FLAN-T5 for text generation
- **Retriever**: Custom FAISS-based retriever
- **Memory**: In-memory chat history
- **RAG Chain**: Combines retriever + prompt + LLM

**Flow**: Query → Retrieve → Format → Generate → Answer

---

##  Getting Started Checklist

- [ ] Clone the repository
- [ ] Create a virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Review `config/config.yaml`
- [ ] Run `python main.py`
- [ ] Test with sample questions
- [ ] Customize with your own Q&A data
- [ ] Share your improvements!

---

##  Contributing

This project is open for learning-based contributions such as:

- Improving prompts
- Trying different embedding models
- Adding a simple UI (Streamlit/FastAPI)
- Improving documentation

---

##  License

This project is licensed under the LICENSE file included in the repository. See [LICENSE](LICENSE) for details.

---

##  FAQ

**Q: Can I use a GPU to speed this up?**
A: Yes! Change `device: cpu` to `device: cuda` in `config.yaml` if you have NVIDIA GPU.

**Q: Can I use my own Q&A data?**
A: Yes! Replace `data/qa_dataset.csv` with your data (must have `question` and `answer` columns).

**Q: What if I want a different language model?**
A: Modify the `model` parameter in `main.py`. Any Hugging Face model works!

**Q: How do I deploy this?**
A: See optional deployment guides: Docker, Streamlit UI, FastAPI backend.

**Q: Is this production-ready?**
A: The core functionality is good, but consider adding error handling, logging, and testing before production.

---

##  Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models)

---

##  Next Steps

1. Replace the dataset with your own domain data
2. Experiment with different chunk sizes
3. Swap the LLM or embedding model
4. Add persistent memory or a web interface

---

Built as a learning-focused RAG project using open-source tools.
Happy experimenting! 🤖✨
