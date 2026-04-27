import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import os

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

# extracting url id form the youtube link
def extract_video_id(url):
  
    url = url.split("?si=")[0]
    
    if "youtu.be" in url:
        return url.split("/")[-1]
    
    parsed = urlparse(url)
    return parse_qs(parsed.query).get("v", [None])[0]

# embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()


st.title("🎬 YouTube Video Chatbot")

url = st.text_input("Please Enter a YouTube URL")

if st.button("Analyze Video"):
    if not url:
        st.warning("Please Enter a YouTube URL!")
    else:
        video_id = extract_video_id(url)

        with st.spinner("Yupp!! Its loadingg..."):
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript_list = ytt_api.fetch(video_id)
                transcript = " ".join(chunk.text for chunk in transcript_list)
                st.success("Transcript is fetched ✅")

            except TranscriptsDisabled:
                st.error("This video doesn't have transcripts available.")
                st.stop()

        with st.spinner("Data is getting processed..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.create_documents([transcript])
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            st.success("Ready! Ask questions!")

# Question section
if "retriever" in st.session_state:
    question = st.text_input("Ask anything about the video")

    if st.button("Ask"):
        if not question:
            st.warning("Please enter a question!")
        else:
            llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7, api_key=groq_api_key)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer only from the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context:
                {context}

                Question: {question}
                """,
                input_variables=["context", "question"]
            )

            with st.spinner("Please wait for the answer..."):
                retrieved_docs = st.session_state.retriever.invoke(question)
                content_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
                final_prompt = prompt.invoke({"context": content_text, "question": question})
                response = llm.invoke(final_prompt)

            st.markdown("### Answer:")
            st.write(response.content)