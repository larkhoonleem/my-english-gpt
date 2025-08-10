import os
import re
import glob
import warnings
import streamlit as st
import smtplib
import random
from email.message import EmailMessage
from typing import List

from openai import OpenAI
import tiktoken

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

GENERATION_MODEL = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_ARGS = dict(temperature=0.0, max_tokens=500)

# ------------------ API Setup ------------------
def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") 
    
API_KEY = get_api_key()
client = OpenAI(api_key=API_KEY)

# ------------------ Tokenizer ------------------
def _get_encoding():
    try:
        return tiktoken.get_encoding("o200k_base")
    except:
        return tiktoken.get_encoding("cl100k_base")

def split_text_by_tokens(text: str, max_tokens: int = 300) -> List[str]:
    enc = _get_encoding()
    lines = [re.sub(r"^\d+(\.\d+)*\s*", "", l.strip())
             for l in text.splitlines() if l.strip() and len(l.strip()) >= 20]
    chunks, cur, cur_tokens = [], [], 0
    for line in lines:
        t = len(enc.encode(line))
        if cur_tokens + t <= max_tokens:
            cur.append(line)
            cur_tokens += t
        else:
            if cur:
                chunks.append("\n".join(cur))
            if t > max_tokens:
                ids = enc.encode(line)
                chunks.append(enc.decode(ids[:max_tokens]))
                rest = enc.decode(ids[max_tokens:])
                cur, cur_tokens = ([rest], len(enc.encode(rest))) if rest.strip() else ([], 0)
            else:
                cur, cur_tokens = [line], t
    if cur:
        chunks.append("\n".join(cur))
    return chunks

# ------------------ Random Sentence Function ------------------
@st.cache_data
def load_sentences_from_file(file_path: str) -> List[str]:
    """Load sentences from the sentences.txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file.readlines() if line.strip()]
        return sentences
    except FileNotFoundError:
        st.error(f"❌ Could not find {file_path}. Please make sure the file exists.")
        return []
    except Exception as e:
        st.error(f"❌ Error reading {file_path}: {e}")
        return []

def get_random_sentence(sentences: List[str]) -> str:
    """Get a random sentence from the list"""
    if sentences:
        return random.choice(sentences)
    return "No sentences available."

# ------------------ Load Documents ------------------
def load_single_document(file_path: str) -> List[Document]:
    """Load a single document based on its file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif file_extension in [".txt", ".text"]:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        else:
            st.warning(f"Unsupported file type: {file_extension} for file {file_path}")
            return []
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def load_and_chunk_documents(folder: str, max_tokens: int = 300):
    """Load and chunk documents from multiple file formats"""
    all_docs = []
    
    # Supported file patterns
    file_patterns = [
        os.path.join(folder, "*.pdf"),
        os.path.join(folder, "*.docx"),
        os.path.join(folder, "*.txt"),
        os.path.join(folder, "*.text")
    ]
    
    # Load all supported files
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            #st.info(f"Loading: {os.path.basename(file_path)}")
            docs = load_single_document(file_path)
            if docs:
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata.update({
                        "source_file": os.path.basename(file_path),
                        "file_type": os.path.splitext(file_path)[1].lower()
                    })
                all_docs.extend(docs)
    
    if not all_docs:
        st.warning(f"No supported documents found in {folder}")
        return []
    
    st.success(f"Loaded {len(all_docs)} document pages from English Knowledge Database")
    
    # Chunk all documents
    all_chunks = []
    for doc in all_docs:
        chunks = split_text_by_tokens(doc.page_content, max_tokens)
        for chunk_text in chunks:
            # Create new document with chunk and preserve metadata
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk_doc)
    
    #st.info(f"Created {len(all_chunks)} text chunks")
    return all_chunks

def build_vectorstore(chunks: List[Document]):
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=API_KEY)
    return FAISS.from_documents(documents=chunks, embedding=embedding)

@st.cache_resource(show_spinner=False)
def get_retriever(folder_path: str, max_tokens_per_chunk: int = 300):
    chunks = load_and_chunk_documents(folder_path, max_tokens_per_chunk)
    vs = build_vectorstore(chunks)
    return vs.as_retriever(search_kwargs={"k": 5})

# ------------------ RAG Query ------------------
def run_rag_query(query: str, retriever):
    retrieved = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    messages = [
        {"role": "system", "content":
         ("You are an experienced English tutor who specializes in teaching English to company executives. "
          "Explain in a respectful, clear, and simple way.\n"
          "Only use the info below.\n"
          "If no answer can be found, say:\n“No relevant information can be found.”")},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
    response = client.chat.completions.create(model=GENERATION_MODEL, messages=messages, **GENERATION_ARGS)
    return response.choices[0].message.content

    # messages = [
    #     {"role": "system", "content":
    #      ("You are an experienced English tutor who specializes in teaching English to company executivies. "
    #       "Explain in a respectful, clear, and simple way.\n"
    #       "Only use the info below.\n"
    #       "If no answer can be found, say:\n“No relevant information can be found.”")},
    #     {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    # ]


# ------------------ Email Upload ------------------
EMAIL_SENDER = "larkhoon.leem@gmail.com"
EMAIL_RECEIVER = "larkhoon.leem@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email_with_attachment(file):
    msg = EmailMessage()
    msg["Subject"] = "File from English GPT"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A new file has been uploaded.")
    msg.add_attachment(file.read(), maintype="application", subtype="octet-stream", filename=file.name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        st.success(f"📨 Successfully sent '{file.name}' for my English Knowledge Collection!")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")

# ------------------ UI ------------------
st.markdown("<h1 style='font-size:36px;'>My English GPT v0.1</h1>", unsafe_allow_html=True)

# ------------------ Random Sentence Section ------------------
st.markdown("### 📝 English Article Practice")

# Load sentences from file
sentences_file = "./sentences.txt"  # Adjust path as needed
sentences = load_sentences_from_file(sentences_file)

if sentences:
    st.info(f"📚 Loaded {len(sentences)} practice sentences")
    
    # Initialize session state for current sentence
    if 'current_sentence' not in st.session_state:
        st.session_state.current_sentence = get_random_sentence(sentences)
    
    # Large button to get random sentence
    if st.button("🎲 Get Random Sentence", 
                 help="Click to get a random sentence for article practice",
                 use_container_width=True):
        st.session_state.current_sentence = get_random_sentence(sentences)
    
    # Display current sentence in a nice box
    st.markdown("#### Current Practice Sentence:")
    st.markdown(f"""
    <div style="
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #4CAF50;
        margin: 10px 0;
    ">
        <h3 style="color: #333; margin: 0;">{st.session_state.current_sentence}</h3>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("⚠️ No sentences loaded. Make sure 'sentences.txt' exists in your directory.")

st.markdown("---")

# ------------------ File Upload Section ------------------
uploaded_file = st.file_uploader("📁 Submit a file to update my English Exercises", type=["pdf", "txt", "docx"])
if uploaded_file:
    send_email_with_attachment(uploaded_file)

# ------------------ Chat Section ------------------
documents_folder = "./documents"
retriever = get_retriever(documents_folder)

st.markdown("### 💬 Ask Questions")
if query := st.chat_input("Ask me anything about English:"):
    st.chat_message("user").write(query)
    answer = run_rag_query(query, retriever)
    st.chat_message("assistant").write(answer)