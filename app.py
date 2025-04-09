import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from GroqEmbeddings
from PIL import Image
import io
import glob

# Load environment variables
load_dotenv()

# Initialize Groq model
def get_groq_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

# Create vector database from tourism.txt
@st.cache_resource
def create_vectordb():
    try:
        # Load the document
        loader = TextLoader("tourism.txt")
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        # Create vector database with HuggingFace embeddings instead of Groq
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        vectordb = FAISS.from_documents(chunks, embeddings)
        
        return vectordb
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None

# Function to get image paths for a location
def get_images_for_location(location):
    location = location.lower()
    image_paths = glob.glob(f"images/{location}*.jpg") + glob.glob(f"images/{location}*.png")
    return image_paths

# Set up page configuration
st.set_page_config(
    page_title="Himachal Pradesh Tourism Guide",
    page_icon="🏔️",
    layout="wide"
)

# Main app
def main():
    st.title("🏔️ Himachal Pradesh Tourism Guide")
    st.write("Ask me anything about Himachal Pradesh tourism!")
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize conversational chain
    if "conversation" not in st.session_state:
        try:
            vectordb = create_vectordb()
            if vectordb:
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                st.session_state.conversation = ConversationalRetrievalChain.from_llm(
                    llm=get_groq_llm(),
                    retriever=vectordb.as_retriever(),
                    memory=memory
                )
        except Exception as e:
            st.error(f"Error initializing conversation: {e}")
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display images for locations if available
            if message["role"] == "assistant" and "location_images" in message:
                images = message["location_images"]
                if images:
                    cols = st.columns(min(3, len(images)))
                    for i, img_path in enumerate(images):
                        try:
                            with cols[i % 3]:
                                st.image(img_path, caption=os.path.basename(img_path).split('.')[0])
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
    
    # User input
    user_input = st.chat_input("Ask about Himachal Pradesh tourism...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Check if the question is about a specific location
        lower_input = user_input.lower()
        locations = ["shimla", "manali", "dharamshala", "mcleodganj", "kullu", "kasol", 
                    "dalhousie", "khajjiar", "chamba", "spiti", "kinnaur", "bir billing"]
        mentioned_location = next((loc for loc in locations if loc in lower_input), None)
        
        # Generate response
        try:
            if "conversation" in st.session_state:
                if "location" in lower_input or "place" in lower_input:
                    # Add specific prompt for location information
                    response = st.session_state.conversation.invoke({
                        "question": f"{user_input} Please provide detailed information about this location in Himachal Pradesh."
                    })
                elif "season" in lower_input or "weather" in lower_input:
                    # Add specific prompt for seasonal recommendations
                    response = st.session_state.conversation.invoke({
                        "question": f"{user_input} Please recommend places based on the season mentioned and explain why they're good for that season."
                    })
                elif "budget" in lower_input or "cost" in lower_input or "low cost" in lower_input:
                    # Add specific prompt for budget recommendations
                    response = st.session_state.conversation.invoke({
                        "question": f"{user_input} Please recommend budget-friendly options and tips for saving money while traveling in Himachal Pradesh."
                    })
                else:
                    response = st.session_state.conversation.invoke({"question": user_input})
                
                ai_response = response["answer"]
            else:
                ai_response = "I'm sorry, but I couldn't access the tourism information. Please check if the tourism.txt file is available."
            
            # Add AI message to chat
            response_obj = {"role": "assistant", "content": ai_response}
            
            # Add images if location is mentioned
            if mentioned_location:
                location_images = get_images_for_location(mentioned_location)
                if location_images:
                    response_obj["location_images"] = location_images
            
            st.session_state.messages.append(response_obj)
            
            # Display AI message
            with st.chat_message("assistant"):
                st.write(ai_response)
                
                # Display images if available
                if mentioned_location:
                    location_images = get_images_for_location(mentioned_location)
                    if location_images:
                        cols = st.columns(min(3, len(location_images)))
                        for i, img_path in enumerate(location_images):
                            try:
                                with cols[i % 3]:
                                    st.image(img_path, caption=os.path.basename(img_path).split('.')[0])
                            except Exception as e:
                                st.error(f"Error displaying image: {e}")
                    
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error: {str(e)}"})

# Sidebar for additional information
def sidebar():
    st.sidebar.title("About")
    st.sidebar.info("This is a Himachal Pradesh Tourism Guide chatbot. Ask questions about tourist places, seasonal recommendations, budget-friendly options, and more!")
    
    st.sidebar.title("Popular Destinations")
    destinations = {
        "Shimla": "The summer capital of British India with colonial architecture.",
        "Manali": "Adventure hub with stunning mountain views.",
        "Dharamshala": "Home to the Dalai Lama and Tibetan culture.",
        "Kullu": "Known for its beautiful valleys and Dussehra festival.",
        "Dalhousie": "Hill station with peaceful ambiance and Scottish architecture."
    }
    
    for place, desc in destinations.items():
        st.sidebar.subheader(place)
        st.sidebar.write(desc)

if __name__ == "__main__":
    sidebar()
    main()