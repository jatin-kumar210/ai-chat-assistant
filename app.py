import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GET HUGGING FACE TOKEN
# Works on both:
# 1. Local computer -> .env
# 2. Streamlit Cloud -> Secrets
# =========================================================

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    except Exception:
        HF_TOKEN = None


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="GenAI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e293b
        );
        color: white;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 45px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 5px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #94a3b8;
        margin-bottom: 35px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #020617;
        border-right: 1px solid #1e293b;
    }

    /* User message */
    .user-message {
        background: #2563eb;
        padding: 16px 20px;
        border-radius: 18px 18px 5px 18px;
        margin: 15px 0 15px auto;
        max-width: 75%;
        color: white;
        font-size: 16px;
    }

    /* AI message */
    .ai-message {
        background: #1e293b;
        border: 1px solid #334155;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 5px;
        margin: 15px auto 15px 0;
        max-width: 75%;
        color: #e2e8f0;
        font-size: 16px;
    }

    /* Welcome card */
    .welcome-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin-bottom: 30px;
    }

    .welcome-card h2 {
        margin-bottom: 10px;
    }

    .welcome-card p {
        color: #94a3b8;
    }

    /* Sidebar title */
    .sidebar-title {
        font-size: 24px;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD AI MODEL
# =========================================================

@st.cache_resource
def load_model():

    # Check token
    if not HF_TOKEN:
        return None

    try:

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=HF_TOKEN
        )

        model = ChatHuggingFace(
            llm=llm
        )

        return model

    except Exception as e:

        st.error(
            f"❌ Model loading failed: {str(e)}"
        )

        return None


# Load model
model = load_model()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🤖 GenAI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 🧠 Model")

    st.info(
        "Qwen 2.5 7B Instruct"
    )

    st.markdown("### 🛠️ Technologies")

    st.write("🐍 Python")
    st.write("🔗 LangChain")
    st.write("🤗 Hugging Face")
    st.write("🎨 Streamlit")

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    st.markdown("---")

    st.caption(
        "Built with ❤️ using Python & Generative AI"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🤖 GenAI Chat Assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Powered by LangChain + Hugging Face'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# TOKEN CHECK
# =========================================================

if not HF_TOKEN:

    st.error(
        "❌ Hugging Face API token not found."
    )

    st.info(
        "For Streamlit Cloud, add HF_TOKEN "
        "inside Settings → Secrets."
    )

    st.stop()


# =========================================================
# MODEL CHECK
# =========================================================

if model is None:

    st.error(
        "❌ AI model could not be initialized."
    )

    st.stop()


# =========================================================
# WELCOME SCREEN
# =========================================================

if len(st.session_state.messages) == 0:

    st.markdown(
        """
        <div class="welcome-card">

            <h2>👋 Welcome to GenAI Chat Assistant</h2>

            <p>
            Ask questions, learn concepts, generate ideas,
            or simply have a conversation with AI.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        st.markdown(
            f"""
            <div class="user-message">

                <b>🧑 You</b>

                <br><br>

                {message["content"]}

            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="ai-message">

                <b>🤖 AI</b>

                <br><br>

                {message["content"]}

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "💬 Ask something..."
)


# =========================================================
# GENERATE AI RESPONSE
# =========================================================

if question:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="user-message">

            <b>🧑 You</b>

            <br><br>

            {question}

        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # GENERATE RESPONSE
    # -----------------------------------------------------

    with st.spinner("🤔 AI is thinking..."):

        try:

            result = model.invoke(question)

            answer = result.content

        except Exception as e:

            answer = (
                f"❌ Error while generating response:\n\n"
                f"{str(e)}"
            )


    # -----------------------------------------------------
    # SAVE AI RESPONSE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # -----------------------------------------------------
    # DISPLAY AI RESPONSE
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="ai-message">

            <b>🤖 AI</b>

            <br><br>

            {answer}

        </div>
        """,
        unsafe_allow_html=True
    )