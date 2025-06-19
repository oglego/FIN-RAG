# FIN-RAG

FIN-RAG is a Retrieval-Augmented Generation (RAG) assistant for financial documents. It allows users to upload financial text files (such as 10-Ks, earnings reports, or market summaries), processes them into searchable chunks, and enables users to ask questions about the content using a language model.

## Features

- **Document Chunking:** Efficiently splits large financial documents into manageable chunks using a Rust-based utility.
- **Embeddings:** Uses HuggingFace sentence-transformers to embed document chunks for semantic search.
- **Retrieval-Augmented Generation:** Combines document retrieval with a language model (Flan-T5) to answer user questions with references to source content.
- **Streamlit UI:** Simple web interface for uploading documents and querying their content.

## Requirements

- Python 3.8+
- Rust (for the chunking utility)
- [Streamlit](https://streamlit.io/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Rayon](https://crates.io/crates/rayon) (Rust dependency)

## Setup

### 1. Clone the Repository

```sh
git clone https://github.com/oglego/FIN-RAG.git
cd fin-rag
```

### 2. Build the Rust Chunking Utility

```sh
cd rust_chunk_text
cargo build --release
cd ..
```

### 3. Install Python Dependencies

```sh
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```sh
streamlit run app.py
```

## Usage

1. Open the Streamlit app in your browser.
2. Upload a `.txt` financial document.
3. Ask questions about the document in natural language.
4. View answers and the relevant source document excerpts.

## File Structure

```
FIN-RAG/
├── app.py                # Main Streamlit application
├── rust_chunk_text/      # Rust utility for chunking text files
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── ...
```

## License

MIT License

## Acknowledgements

- [HuggingFace Transformers](https://huggingface.co/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
-