import streamlit as st
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from datetime import datetime

# Streamlit app
def main():
    # Initialize embedding and language model
    EMBEDDING = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="mixtral-8x7b-32768"
    )

    # Load the FAISS vector embeddings 
    vectors = FAISS.load_local(
        "faiss-index", 
        embeddings=EMBEDDING, 
        allow_dangerous_deserialization=True
    )

    # Set up prompt template and QA chain
    prompt = ChatPromptTemplate.from_template(
        """
        You have expertise in the Indian Constitution and the laws, You have to answer some related questions!
     
        Try to respond with the latest information available. First, try to make a 
        sensible response from the data provided through the vector database of India's constitution.
        If you don't know the answer, respond accordingly.
        Keep the responses relevant to the Constitution of India pdf only by first searching the latest information through the database!
        {context}
        Question: {question}
        """
    )

    # Set up conversation memory
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = st.session_state["chat_memory"]

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectors.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
        memory=memory,  
        output_key="result"
    )

    # Sidebar for navigation
    page = st.sidebar.radio("Select a Page", ["Main Chatbot", "Chat History"])

    if page == "Main Chatbot":
        st.title("📜 Indian Constitution Chatbot")
        st.write("Ask any question about the Indian Constitution!")

        # Use a form to wrap the input and submit button together
        with st.form(key="qa_form"):
            query = st.text_input("Enter your prompt:")
            submit_button = st.form_submit_button("Submit")

        # When submit button is pressed, process the query
        if submit_button and query:
            # Capture the current time of the query submission
            question_time = datetime.now().strftime("%H:%M:%S")
            
            with st.spinner("Processing your query..."):
                # Call the QA chain with the query and memory
                result = qa_chain({"query": query})
                # Display the answer
                st.write("### Answer:")
                st.write(result['result'])

    elif page == "Chat History":
        st.title("📜 Chat History")
        st.write("Here is the chat history with the Indian Constitution chatbot:")

        # Display chat history from the memory
        for message in memory.chat_memory.messages:
            if message.type == "human":
                st.write(f"**You:** {message.content}")
            elif message.type == "ai":
                st.write(f"**Bot:** {message.content}")


if __name__ == "__main__":
    load_dotenv()
    main()
