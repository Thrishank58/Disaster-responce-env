import streamlit as st
from pdf_reader import read_pdf
from ocr_reader import read_image
from llm import explain_text, followup_chat

# ---- Load CSS ----
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.set_page_config(page_title="Medical AI Tutor", layout="wide")

st.title("🩺 Medical AI Tutor")
st.warning("⚠️ Educational use only. Not for diagnosis or treatment.")

# ---- Session memory ----
if "context" not in st.session_state:
    st.session_state.context = ""

# ---- Layout ----
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📄 Upload File")
    uploaded_file = st.file_uploader(
        "Upload PDF or Image",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    explain_btn = st.button("🧠 Explain Like a Tutor")

with col2:
    st.subheader("📚 Explanation")

    if uploaded_file and explain_btn:
        with st.spinner("AI tutor is explaining…"):
            if uploaded_file.type == "application/pdf":
                text = read_pdf(uploaded_file)
            else:
                text = read_image(uploaded_file)

            explanation = explain_text(text)
            st.session_state.context = explanation
            st.write(explanation)

# ---- Follow-up chat ----
if st.session_state.context:
    st.markdown("---")
    st.subheader("💬 Ask a Follow-up Question")

    question = st.text_input("Ask anything about this topic")

    if st.button("Ask"):
        with st.spinner("Thinking like a tutor…"):
            answer = followup_chat(st.session_state.context, question)
        st.write(answer)
