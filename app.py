import streamlit as st
import subprocess

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def preprocess_financial_text(text):
    """
    Preprocess financial text by replacing specific terms with standardized labels.

    This makes embeddings and retrieval more effective by ensuring consistent terminology.
    """
    replacements = {
        "foreign currency": "foreign_currency",
        "international operations": "international_operations",
        "currency risk": "currency_risk"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def rust_chunk_file(file_path, chunk_size=500):
    """
    Splits a file into chunks using a custom external Rust binary.

    Args:
        file_path (str): The path to the file to be chunked.
        chunk_size (int, optional): The desired size of each chunk. Defaults to 500.

    Returns:
        list of str: A list containing the text chunks extracted from the file.
    """
    result = subprocess.run(
        ["rust_chunk_text/target/release/rust_chunk_text", file_path, str(chunk_size)],
        capture_output=True,
        text=True
    )
    chunks = result.stdout.split("===CHUNK===")
    return [c.strip() for c in chunks if c.strip()]

@st.cache_resource
def embed_docs(_docs):
    """
    Embeds documents using a HuggingFace model and creates a FAISS vector store retriever.

    Args:
        _docs (list): A list of documents to embed.

    Returns:
        Retriever: A retriever object for similarity search with the embedded documents.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def load_llm():
    """
    Loads a HuggingFace Flan-T5 model and tokenizer, and creates a text2text-generation pipeline.

    Returns:
        HuggingFacePipeline: A pipeline object for text-to-text generation using the loaded model.
    """
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)

# Build RAG Chain 
@st.cache_resource
def build_rag(_retriever, _llm):
    """
    Builds a Retrieval-Augmented Generation (RAG) pipeline using the provided retriever and language model.

    Args:
        _retriever: The retriever object for fetching relevant documents.
        _llm: The language model to generate answers.

    Returns:
        RetrievalQA: A RetrievalQA chain configured with the given retriever and language model.
    """
    return RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=_retriever,
        return_source_documents=True
    )

def main():
    """
    Streamlit UI for the Financial RAG Assistant.

    Allows users to upload a financial text document, processes it into chunks,
    embeds the chunks, and sets up a Retrieval-Augmented Generation (RAG) pipeline.

    Users can then ask questions about the uploaded document and receive answers
    with source document references.
    """

    st.title("Financial RAG Assistant")
    st.markdown("Ask questions about your financial documents like 10-Ks, earnings reports, or market summaries.  This RAG" \
    " assistant preprocesses text and is focused on financial documents, particularly risk factors and international operations.")

    uploaded_file = st.file_uploader("Upload a .txt financial document", type=["txt"])

    if uploaded_file:
        # Save uploaded file
        with open("temp_doc.txt", "wb") as f:
            f.write(uploaded_file.read())

        raw_chunks = rust_chunk_file("temp_doc.txt")
        docs = [Document(page_content=preprocess_financial_text(c), metadata={"section": "Risk Factors"}) for c in raw_chunks]
        retriever = embed_docs(docs)
        llm = load_llm()
        rag_chain = build_rag(retriever, llm)

        user_query = st.text_input("Ask a financial question:")
        if user_query:
            result = rag_chain({"query": user_query})
            st.subheader("Answer:")
            st.write(result["result"])

            with st.expander("Source Documents"):
                for doc in result["source_documents"]:
                    st.markdown("---")
                    st.markdown(doc.page_content[:500] + "...")
                    
if __name__ == "__main__":
    main()
else:
    st.error("Please upload a valid .txt file.")
    st.stop()
