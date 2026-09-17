
import streamlit as st
import json
import faiss
import os

from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Al Shifa Hospital Assistant",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #F4F8FC;
    }

    /* Header */
    .main-header {
        background: linear-gradient(
            135deg,
            #0F2747,
            #1976D2
        );

        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        margin: 0;
        font-size: 32px;
    }

    .main-header p {
        margin-top: 8px;
        font-size: 16px;
        opacity: 0.9;
    }

    /* Cards */
    .info-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #E2E8F0;
        margin-bottom: 15px;
    }

    /* User message */
    .user-message {
        background-color: #E3F2FD;
        padding: 15px 18px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #1976D2;
    }

    /* Assistant message */
    .assistant-message {
        background-color: white;
        padding: 18px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #00A6A6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Suggested questions */
    .suggestion-title {
        font-size: 20px;
        font-weight: 700;
        color: #0F2747;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F2747;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        margin-top: 30px;
        padding: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE PATHS
# ============================================================

INDEX_PATH = "faiss_index/hospital.index"
CHUNKS_PATH = "faiss_index/chunks.json"
METADATA_PATH = "faiss_index/metadata.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    return faiss.read_index(INDEX_PATH)


index = load_faiss_index()


# ============================================================
# LOAD CHUNKS
# ============================================================

@st.cache_data
def load_chunks():

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


chunks = load_chunks()


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


metadata = load_metadata()


# ============================================================
# GROQ CLIENT
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:

    st.error(
        "Groq API key was not found. "
        "Please connect GROQ_API_KEY in Colab."
    )

    st.stop()


client = Groq(
    api_key=GROQ_API_KEY
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(
    question,
    top_k=3
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            {
                "score": float(score),
                "text": chunks[idx],
                "metadata": metadata[idx]
            }
        )

    return results


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    results
):

    context = "\n\n".join(
        [
            f"Source: {result['metadata']['source']}\n"
            f"{result['text']}"
            for result in results
        ]
    )

    prompt = f"""
You are an AI knowledge assistant for Al Shifa Hospital.

Your job is to answer questions using ONLY the hospital
knowledge provided below.

IMPORTANT RULES:

1. Use only the provided hospital knowledge.
2. Do not invent information.
3. Do not make up hospital policies, procedures,
   departments, timings, phone numbers, or medical information.
4. If the answer is not available in the provided knowledge,
   clearly say that the information is not available in the
   hospital knowledge base.
5. Give a clear and professional answer.
6. Do not mention FAISS, embeddings, or the retrieval process.

HOSPITAL KNOWLEDGE:
-------------------
{context}
-------------------

USER QUESTION:
{question}

Answer the user's question clearly and concisely.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <h1>🏥 Al Shifa Hospital Assistant</h1>

        <p>
        Ask questions about hospital services, admission,
        emergency procedures, departments, patient safety,
        and hospital policies.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <h2>🏥 Al Shifa Hospital</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📚 Knowledge Areas")

    st.markdown(
        """
        • 🏨 Hospital Information  
        • 📝 Admission  
        • 🚑 Emergency  
        • 🫀 Departments  
        • 🛡️ Patient Safety
        """
    )

    st.markdown("---")

    st.markdown("### 🔎 RAG Assistant")

    st.write(
        "This assistant answers questions "
        "using the hospital knowledge base."
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# WELCOME MESSAGE
# ============================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        """
        <div class="info-card">

        <h3>👋 Welcome!</h3>

        <p>
        I am the Al Shifa Hospital knowledge assistant.
        You can ask me questions about hospital procedures,
        admission, emergency care, departments, and patient safety.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.markdown(
    '<div class="suggestion-title">💡 What can I ask?</div>',
    unsafe_allow_html=True
)

st.write(
    "Click a question below to ask the assistant:"
)


suggested_questions = [

    "What are the patient identification requirements?",

    "What is the inpatient admission process?",

    "What are the emergency triage procedures?",

    "What are the infection prevention procedures?",

    "What are the medication safety rules?",

    "What is the hospital privacy policy?"

]


selected_question = None


col1, col2 = st.columns(2)


with col1:

    if st.button(
        "👤 Patient identification requirements",
        key="q1",
        use_container_width=True
    ):

        selected_question = suggested_questions[0]


    if st.button(
        "🏥 Inpatient admission process",
        key="q2",
        use_container_width=True
    ):

        selected_question = suggested_questions[1]


    if st.button(
        "🚑 Emergency triage procedures",
        key="q3",
        use_container_width=True
    ):

        selected_question = suggested_questions[2]


with col2:

    if st.button(
        "🛡️ Infection prevention procedures",
        key="q4",
        use_container_width=True
    ):

        selected_question = suggested_questions[3]


    if st.button(
        "💊 Medication safety rules",
        key="q5",
        use_container_width=True
    ):

        selected_question = suggested_questions[4]


    if st.button(
        "🔐 Hospital privacy policy",
        key="q6",
        use_container_width=True
    ):

        selected_question = suggested_questions[5]


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">

            👤 <b>You</b><br><br>

            {message["content"]}

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="assistant-message">

            🤖 <b>Al Shifa Assistant</b><br><br>

            {message["content"]}

            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        if "sources" in message:

            with st.expander(
                f"📚 View sources ({len(message['sources'])})"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"📄 `{source}`"
                    )


# ============================================================
# HANDLE SUGGESTED QUESTION
# ============================================================

if selected_question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": selected_question
        }
    )

    results = retrieve_documents(
        selected_question,
        top_k=3
    )

    answer = generate_answer(
        selected_question,
        results
    )

    sources = [
        result["metadata"]["source"]
        for result in results
    ]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

    st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Al Shifa Hospital..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    results = retrieve_documents(
        question,
        top_k=3
    )

    answer = generate_answer(
        question,
        results
    )

    sources = [
        result["metadata"]["source"]
        for result in results
    ]

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )

    st.rerun()


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="footer">

    🏥 Al Shifa Hospital Knowledge Assistant

    <br>

    This assistant provides information from the hospital
    knowledge base and does not replace professional medical
    advice or emergency care.

    </div>
    """,
    unsafe_allow_html=True
)
