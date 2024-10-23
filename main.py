import streamlit as st
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


# Streamlit app
def main():
    # Initialize embedding and language model
    EMBEDDING = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

    # Load the FAISS vector embeddings 
    vectors = FAISS.load_local(
        "faiss_index", 
        embeddings=EMBEDDING, 
        allow_dangerous_deserialization=True
    )

    # Set up prompt template and QA chain
    prompt = ChatPromptTemplate.from_template(
        """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, say that you don't know it.
        Keep the answers relevant to the constitution of India pdf only.
        {context}
        Question: {question}
        """
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectors.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    st.title("Indian Constitution Chatbot")
    
    st.write("Ask any question about the Indian Constitution!")

    # Use a form to wrap the input and submit button together
    with st.form(key="qa_form"):
        query = st.text_input("Enter your prompt:")
        submit_button = st.form_submit_button("Submit")

    # When submit button is pressed, process the query
    if submit_button and query:
        with st.spinner("Processing your query..."):
            result = qa_chain({"query": query})
            st.write("Answer:", result['result'])


if __name__ == "__main__":
    load_dotenv()
    main()
