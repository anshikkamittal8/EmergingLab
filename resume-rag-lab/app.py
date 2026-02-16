import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env
load_dotenv()

# -----------------------------
# Utility Functions
# -----------------------------

def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def create_vector_db(text):
    # Split the Job Description into chunks for better context retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    return vector_db

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume & Internship Helper")
st.markdown("---")

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Section")
    resume_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
    jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")

with col2:
    st.subheader("💡 Instructions")
    st.info("""
    1. Upload your latest Resume.
    2. Upload the PDF version of the Job Description.
    3. Click 'Analyze' to see how well you match!
    """)

if st.button("🚀 Analyze Resume"):
    if resume_file and jd_file:
        try:
            with st.spinner("Processing documents..."):
                resume_text = read_pdf(resume_file)
                jd_text = read_pdf(jd_file)

            if not resume_text or not jd_text:
                st.error("Could not extract text from one of the files.")
                st.stop()

            with st.spinner("Analyzing requirements..."):
                # RAG: Store JD in FAISS so the LLM can query specific requirements
                vector_db = create_vector_db(jd_text)
                retriever = vector_db.as_retriever(search_kwargs={"k": 3})

                llm = ChatOpenAI(model="gpt-4o", temperature=0.2) # Using 4o for better analysis

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    chain_type="stuff"
                )

                # RAG Prompt: The LLM gets the Resume as global context 
                # and the JD via the retriever
                query = f"""
                You are a Senior Technical Recruiter. Compare the following Resume text against the retrieved Job Description:
                
                RESUME TEXT:
                {resume_text}
                
                Provide a detailed report:
                1. MATCH PERCENTAGE: A score out of 100.
                2. KEY STRENGTHS: What the candidate already has.
                3. CRITICAL GAPS: Specific technical or soft skills missing.
                4. RESUME REWRITING TIPS: How to describe existing experience to better match this JD.
                5. CAREER PATH: 2 other roles this candidate is qualified for.
                """

                result = qa_chain.invoke(query)

            st.success("Analysis Complete!")
            st.markdown("### 📊 Recommendation Report")
            st.markdown(result["result"])
            
        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please upload both documents to proceed.")