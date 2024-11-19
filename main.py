import streamlit as st
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from datetime import datetime
from gtts import gTTS
import io
import os

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
        Keep the responses Concise (of a maximum of 2 to 5 lines generally) unless asked to elongate the result.
        Try to respond with the latest information available. First, try to make a 
        sensible response from the data provided through the vector database of India's constitution.
        If you don't know the answer, respond accordingly. Remember that the latest amendment done in the constitution was 106th in 2023, and 
        India has 448 articles, 25 parts, and 12 schedules yet in the Constitution.
        Keep the responses relevant to the Constitution of India pdf only by searching for the latest information in the database!
        {context}
        Question: {question}
        """
    )

    # Set up conversation memory
    if "chat_memory" not in st.session_state:
        st.session_state["chat_memory"] = {
            "messages": []
        }
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None  # To store the last response

    memory = st.session_state["chat_memory"]

    # Create a retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectors.as_retriever(),
        chain_type_kwargs={"prompt": prompt},
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

        # Display the latest response if it exists
        if st.session_state.get("last_response") and not submit_button:
            st.write("### Answer:")
            st.write(st.session_state["last_response"])

            # Display the latest audio if it exists
            if st.session_state.get("audio_bytes"):
                st.markdown("#### Your audio response:")
                st.audio(st.session_state["audio_bytes"], format="audio/mp3", start_time=0)

        # When submit button is pressed, process the query
        if submit_button and query:
            with st.spinner("Processing your query..."):
                # Call the QA chain with the query
                result = qa_chain({"query": query})
                response = result['result']

                # Store the query and response in memory with timestamps
                memory["messages"].append({"type": "human", "content": query, "timestamp": datetime.now()})
                memory["messages"].append({"type": "ai", "content": response, "timestamp": datetime.now()})

                # Store the response in session_state for persistence
                st.session_state["last_response"] = response

                # Display the answer
                st.write("### Answer:")
                st.write(response)

            with st.spinner("Converting the response to audio..."):
                # Convert response to speech
                try:
                    tts = gTTS(text=response, lang='en')
                    # Save to a BytesIO object
                    audio_bytes = io.BytesIO()
                    tts.write_to_fp(audio_bytes)  # Correct method to write directly to BytesIO
                    audio_bytes.seek(0)  # Move to the beginning of the BytesIO object

                    # Store audio bytes in session_state
                    st.session_state["audio_bytes"] = audio_bytes

                    # Display audio in Streamlit
                    st.markdown("#### Your audio response:")
                    st.audio(audio_bytes, format="audio/mp3", start_time=0)
                except Exception as e:
                    st.error(f"Failed to generate audio: {e}")


    elif page == "Chat History":
        st.title("📜 Chat History")

        # Group messages into prompt-response pairs for better readability
        grouped_messages = []
        temp_pair = {}
    
        for message in memory["messages"]:
            if message["type"] == "human":
                # Start a new pair
                temp_pair = {"prompt": message}
            elif message["type"] == "ai" and temp_pair:
                # Complete the pair
                temp_pair["response"] = message
                grouped_messages.append(temp_pair)
                temp_pair = {}
    
        # Display chat history
        for pair in grouped_messages[::-1]:  # Reverse to display latest first
            prompt_time = pair["prompt"]["timestamp"].strftime("%H:%M:%S")
            response_time = pair["response"]["timestamp"].strftime("%H:%M:%S")
            st.write(f"**PROMPT ({prompt_time}):** {pair['prompt']['content']}")
            st.write(f"**RESPONSE ({response_time}):** {pair['response']['content']}")
            st.markdown("---")        
            st.markdown("  \n")

if __name__ == "__main__":
    load_dotenv()
    main()
