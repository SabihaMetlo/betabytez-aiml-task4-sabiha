from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os

LOGO_SVG = """
<svg width="32" height="32" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
  <circle cx="20" cy="20" r="18" fill="none" stroke="#7F77DD" stroke-width="2"/>
  <polygon points="20,8 24,20 20,32 16,20" fill="#7F77DD"/>
  <circle cx="20" cy="20" r="2.5" fill="#0e1117"/>
</svg>
"""

def load_text_with_encoding(filepath):
    encodings = ["utf-8", "utf-16", "cp1252"]
    for enc in encodings:
        try:
            loader = TextLoader(filepath, encoding=enc)
            return loader.load()
        except RuntimeError:
            continue
    raise RuntimeError(f"Could not decode {filepath} with any known encoding")


@st.cache_resource
def load_pipeline():
    taskbook_loader = PyPDFLoader("data/AI-ML-TasksBook - Copy.pdf")
    guidelines_loader = PyPDFLoader("data/Guidelines - Copy.pdf")
    pdf_docs = taskbook_loader.load() + guidelines_loader.load()

    readme_files = ["README.md", "README_2.md", "README_3.md"]
    readme_docs = []
    for filename in readme_files:
        readme_docs.extend(load_text_with_encoding(f"data/{filename}"))

    all_documents = pdf_docs + readme_docs

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embedding_model)
    vector_store.save_local("faiss_index")

    load_dotenv()
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    return vector_store, llm


vector_store, llm = load_pipeline()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "saved_chats" not in st.session_state:
    st.session_state.saved_chats = []
if "active_chat_index" not in st.session_state:
    st.session_state.active_chat_index = None

st.set_page_config(page_title="InternGuide", page_icon="🧭", layout="wide")

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


def render_chat_row(chat, real_index):
    is_active = (st.session_state.active_chat_index == real_index)
    wrapper_class = "recent-active" if is_active else ""
    rename_key = f"renaming_{real_index}"
    if rename_key not in st.session_state:
        st.session_state[rename_key] = False

    col_btn, col_menu = st.columns([6, 1])
    with col_btn:
        st.markdown(f"<div class='{wrapper_class}'>", unsafe_allow_html=True)
        label = ("⭐ " if chat.get("starred") else "") + chat["title"]
        if st.button(label, use_container_width=True, key=f"open_{real_index}"):
            if st.session_state.chat_history and st.session_state.active_chat_index is None:
                fq = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "New chat")
                t = fq if len(fq) <= 30 else fq[:30] + "..."
                st.session_state.saved_chats.insert(0, {"title": t, "history": st.session_state.chat_history, "starred": False})
            st.session_state.chat_history = chat["history"]
            st.session_state.active_chat_index = real_index
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_menu:
        with st.popover("⋮"):
            if st.session_state[rename_key]:
                new_title = st.text_input(
                    "New name", value=chat["title"],
                    key=f"rename_input_{real_index}", label_visibility="collapsed"
                )
                if new_title != chat["title"]:
                    st.session_state.saved_chats[real_index]["title"] = new_title
                    st.session_state[rename_key] = False
                    st.rerun()
            else:
                star_label = "☆ Unstar" if chat.get("starred") else "⭐ Star"
                if st.button(star_label, key=f"star_{real_index}", use_container_width=True):
                    st.session_state.saved_chats[real_index]["starred"] = not chat.get("starred", False)
                    st.rerun()
                if st.button("✏️ Rename", key=f"rename_btn_{real_index}", use_container_width=True):
                    st.session_state[rename_key] = True
                    st.rerun()
                st.markdown("<div class='delete-option'>", unsafe_allow_html=True)
                if st.button("🗑 Delete", key=f"delete_{real_index}", use_container_width=True):
                    st.session_state.saved_chats.pop(real_index)
                    if st.session_state.active_chat_index == real_index:
                        st.session_state.active_chat_index = None
                        st.session_state.chat_history = []
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    logo_col, title_col = st.columns([1, 4])
    with logo_col:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with title_col:
        st.markdown("<div style='padding-top:6px; font-weight:600; font-size:18px;'>InternGuide</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    if st.button("＋  New chat", use_container_width=True):
        if st.session_state.chat_history:
            first_question = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "New chat")
            title = first_question if len(first_question) <= 30 else first_question[:30] + "..."
            st.session_state.saved_chats.insert(0, {"title": title, "history": st.session_state.chat_history, "starred": False})
        st.session_state.chat_history = []
        st.session_state.active_chat_index = None
        st.rerun()

    if st.session_state.saved_chats:
        starred = [c for c in st.session_state.saved_chats if c.get("starred")]
        unstarred = [c for c in st.session_state.saved_chats if not c.get("starred")]

        if starred:
            st.markdown("<div class='sidebar-section-label'>Starred</div>", unsafe_allow_html=True)
            for chat in starred:
                real_index = st.session_state.saved_chats.index(chat)
                render_chat_row(chat, real_index)

        st.markdown("<div class='sidebar-section-label'>Recents</div>", unsafe_allow_html=True)
        for chat in unstarred[:10]:
            real_index = st.session_state.saved_chats.index(chat)
            render_chat_row(chat, real_index)
    else:
        st.markdown("<div class='sidebar-section-label'>Recents</div>", unsafe_allow_html=True)
        st.caption("No previous chats yet")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-footer'>Built by Sabiha Metlo — BetaBytez AI/ML Internship, Task 4</div>",
        unsafe_allow_html=True
    )

# ---------- MAIN CHAT AREA ----------
st.title("InternGuide")
st.caption("Ask questions grounded in the BetaBytez Task Book, Guidelines, and Task 1-3 READMEs")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-msg'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='ai-msg'>{message['content']}</div>", unsafe_allow_html=True)

user_question = st.chat_input("Ask a question...")

if user_question:
    st.markdown(f"<div class='user-msg'>{user_question}</div>", unsafe_allow_html=True)
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    results = vector_store.similarity_search(user_question, k=3)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""Answer the question using ONLY the context below. 
If the answer is not in the context, say "I don't have information about that in the provided documents."
Do not infer numbers, task names, or facts that are not explicitly and clearly stated in the context.

Context:
{context}

Question: {user_question}

Answer:"""
    response = llm.invoke(prompt)

    st.markdown(f"<div class='ai-msg'>{response.content}</div>", unsafe_allow_html=True)
    with st.expander("📄 View source chunks"):
        for i, doc in enumerate(results):
            st.write(f"**Source {i+1}:**")
            st.write(doc.page_content)

    st.session_state.chat_history.append({"role": "assistant", "content": response.content})