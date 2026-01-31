# Skincare Ecommerce FAQ RAG Chatbot 🧴

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot specifically designed to answer skincare and ecommerce product questions using local machine learning models. This project requires **no API keys** and runs entirely on your CPU.

---

## 📋 Table of Contents

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
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🤖 What is This Project?

This is a **RAG (Retrieval-Augmented Generation) Chatbot** built specifically for skincare ecommerce customer support. Instead of relying on large API-based models, this chatbot:

1. **Retrieves** relevant Q&A from your knowledge base using semantic search
2. **Augments** those results into a prompt
3. **Generates** accurate answers using a local language model

**Key Benefits:**
- ✅ No API costs or external service dependencies
- ✅ Privacy-first: Everything runs locally on your machine
- ✅ Fast retrieval using FAISS vector database
- ✅ Beginner-friendly: Well-documented and modular
- ✅ Customizable: Easy to adapt for your own Q&A data

---

## ✨ Features

- 🔍 **Semantic Search**: Uses embeddings to find relevant answers
- 💬 **Conversation Memory**: Maintains chat history within sessions
- 🚀 **Local & CPU-Friendly**: No GPU required, no cloud dependencies
- 📊 **FAISS Vector Database**: Ultra-fast similarity search
- ⚙️ **Configuration-Driven**: Easy to customize via `config.yaml`
- 📚 **LangChain Integration**: Modern LLM framework for building chains

---

## 💻 System Requirements

Before getting started, ensure your system meets these requirements:

### Minimum Requirements:
- **RAM**: 4 GB (8 GB recommended)
- **Storage**: 5-10 GB free space
- **Processor**: Any modern CPU (Works for low end PCs)
- **OS**: Windows, macOS, or Linux
- **Python**: Version 3.10 or higher

### Optional (Not Required):
- GPU for faster inference (GPU not required but helpful)
- Conda package manager (recommended for better dependency management)

---

## 🚀 Installation & Setup

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
- `torch`: Deep learning framework (downloaded automatically)
- `faiss-cpu`: Vector database for fast similarity search
- `langchain` & `langchain-huggingface`: LLM orchestration
- `transformers`: Pre-trained language models
- `pyyaml`: Configuration file handling
- `tqdm`: Progress bars

**Installation Time:** 5-15 minutes (depending on internet speed)

### Step 4: Download Pre-trained Models

The first time you run the project, models are automatically downloaded:
- `paraphrase-MiniLM-L3-v2`: For embeddings (~50 MB)
- `google/flan-t5-base`: For text generation (~990 MB)

These are cached locally, so subsequent runs are instant.

### Step 5: Configure the Project

The project uses `config/config.yaml` for all settings. You can customize:

```yaml
paths:
  input_csv: data/qa_dataset.csv          # Your Q&A data
  output_documents: data/processed/documents.txt
  processed_dir: data/processed
  vector_index: vectordb/faiss.index      # FAISS database

processing:
  question_column: question               # Column name in CSV
  answer_column: answer                   # Column name in CSV

chunking:
  chunk_size: 80                          # How many words per chunk
  overlap: 30                             # Overlap between chunks

embedding:
  model_name: paraphrase-MiniLM-L3-v2    # Embedding model
  device: cpu                             # Use 'cuda' for GPU
  batch_size: 32                          # How many texts to process at once
```

---

## 📁 Project Structure

```
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
```

---

## 🔄 How It Works

Here's the flow of data through the chatbot:

### 1. **Data Preparation Phase** (One-time setup)
```
CSV Dataset → Documents → Chunks → Embeddings → FAISS Index
```

- **CSV to Documents**: Your Q&A pairs are converted to text
- **Chunking**: Long texts are split into smaller pieces (configurable)
- **Embeddings**: Each chunk is converted to a vector using `paraphrase-MiniLM-L3-v2`
- **FAISS Index**: Vectors are stored in a searchable index

### 2. **Query & Response Phase** (During chatbot runtime)
```
User Question → Embedding → Search FAISS → Retrieve Top-3 → Generate Answer
```

- **User Input**: You ask a question about skincare
- **Embedding**: Question is converted to a vector
- **Semantic Search**: FAISS finds 3 most similar chunks (by cosine similarity)
- **Context + LLM**: Results are passed to `google/flan-t5-base` LLM
- **Answer**: LLM generates a natural language response using the context

### 3. **Conversation Memory** (Sessions)
- Each chat session is stored separately
- Memory is in-memory (lost when app stops)
- Easy to extend with database persistence

---

## 💬 Usage

### Running the Chatbot

```bash
python main.py
```

### Example Interaction

```
User: How do I use retinol for sensitive skin?

Chatbot: Based on our knowledge base, retinol is a powerful ingredient 
but can be harsh on sensitive skin. We recommend: 
1. Start with low concentrations (0.1-0.25%)
2. Use 2-3 times per week initially
3. Always apply moisturizer after
4. Use SPF 30+ during the day
```

### Session Management

The chatbot maintains conversation history per session:

```python
# Different sessions maintain separate memory
session_1 = get_session_history("user_123")
session_2 = get_session_history("user_456")
```

---

## 🏗️ Project Components Explained

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
- Creates an index for O(1) similarity search
- Uses Inner Product (equivalent to cosine similarity for normalized vectors)
- ~50,000+ documents searchable in milliseconds

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

## 🛠️ Troubleshooting

### "ImportError: No module named 'sentence_transformers'"
**Solution**: Install requirements again
```bash
pip install -r requirements.txt
```

### "Model download stuck or slow"
**Solution**: Models are cached after first download. Check internet connection.
- First run: 5-15 minutes (downloads ~1 GB)
- Subsequent runs: Instant

### "FAISS index not found"
**Solution**: The index is pre-built. If missing, regenerate it:
```bash
python vectordb/vector_store.py
```

### "Out of memory error"
**Solution**: Reduce batch size in `config.yaml`:
```yaml
embedding:
  batch_size: 8  # Reduce from 32 to 8
```

### "Chatbot gives irrelevant answers"
**Solution**: Check your Q&A dataset. Quality in = quality out. Also try:
- Increasing `top_k` in retriever (search more chunks)
- Adjusting `chunk_size` for better granularity

### "Windows: 'python' is not recognized"
**Solution**: Use full path or add Python to PATH
```bash
C:\Python\python.exe main.py
# OR if Python is in PATH:
python main.py
```

---

## 📖 Getting Started Checklist

- [ ] Clone the repository
- [ ] Create a virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Review `config/config.yaml`
- [ ] Run `python main.py`
- [ ] Test with sample questions
- [ ] Customize with your own Q&A data
- [ ] Share your improvements!

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Make changes** and test thoroughly
4. **Commit**: `git commit -m "Add your feature"`
5. **Push**: `git push origin feature/your-feature`
6. **Create a Pull Request**

### Areas for contribution:
- Better embedding models
- Database persistence for chat history
- Fine-tuned models for skincare domain
- Unit tests and documentation

---

## 📄 License

This project is licensed under the LICENSE file included in the repository. See [LICENSE](LICENSE) for details.

---

## ❓ FAQ

**Q: Can I use a GPU to speed this up?**
A: Yes! Change `device: cpu` to `device: cuda` in `config.yaml` if you have NVIDIA GPU.

**Q: Can I use my own Q&A data?**
A: Yes! Replace `data/qa_dataset.csv` with your data (must have `question` and `answer` columns).

**Q: What if I want a different language model?**
A: Modify the `model` parameter in `main.py`. Any Hugging Face model works!

**Q: How do I deploy this?**
A: See optional deployment guides: Docker, Streamlit UI, FastAPI backend.

**Q: Is this production-ready?**
A: The core functionality is solid, but consider adding error handling, logging, and testing before production.

---

## 📚 Resources

- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Library](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Models](https://huggingface.co/models)

---

## 🎯 Next Steps

1. **Customize**: Add your own skincare Q&A dataset
2. **Improve**: Experiment with different embedding models
3. **Enhance**: Build a web interface with Flask/Streamlit
4. **Deploy**: Deploy using Docker or cloud services

---

## 📞 Support

If you encounter issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review error messages carefully
3. Check that all dependencies are installed
4. Open an issue on GitHub with details of your problem

---

**Happy Chatting! 🤖✨**

Built with ❤️ for the skincare ecommerce community.
