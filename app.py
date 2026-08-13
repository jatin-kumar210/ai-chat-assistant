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
# =========================================================

# Local computer -> .env
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit Cloud -> Secrets
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

    /* Main background */
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
        font-size: 46px;
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

    /* Sidebar title */
    .sidebar-title {
        font-size: 25px;
        font-weight: 700;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(
            135deg,
            #0f172a,
            #1e293b
        );

        border: 1px solid #334155;

        border-radius: 22px;

        padding: 35px;

        text-align: center;

        margin-bottom: 30px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.25);
    }

    .welcome-card h2 {
        font-size: 28px;
        margin-bottom: 12px;
    }

    .welcome-card p {
        color: #94a3b8;
        font-size: 16px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        margin-bottom: 15px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        height: 45px;
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

    st.info("Qwen 2.5 7B Instruct")

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
                Your AI-powered assistant for learning,
                exploring ideas, solving problems,
                and having intelligent conversations.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👨‍💻"
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # AI MESSAGE
    # -----------------------------------------------------

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.markdown(
                message["content"]
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

    with st.chat_message(
        "user",
        avatar="👨‍💻"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # GENERATE AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖"
    ):

        with st.spinner(
            "🤔 AI is thinking..."
        ):

            try:

                result = model.invoke(
                    question
                )

                answer = result.content

            except Exception as e:

                answer = (
                    "❌ Something went wrong.\n\n"
                    f"Error: {str(e)}"
                )

        st.markdown(
            answer
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