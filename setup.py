import os
import shutil
from dotenv import load_dotenv

def setup_project():
    """
    Set up the project directory structure.
    Creates necessary folders and files for the Himachal Tourism Chatbot.
    Loads environment variables from .env if available.
    """
    # Load environment variables from .env
    load_dotenv()

    # Create folders
    os.makedirs("images", exist_ok=True)

    # Create .env.example file for reference (but do not include secrets)
    if not os.path.exists(".env.example"):
        with open(".env.example", "w") as f:
            f.write("""LANGCHAIN_PROJECT=GENAIAPP
HF_TOKEN=your_huggingface_token_here
LANGSMITH_API_KEY=your_langsmith_api_key_here
USER_AGENT=your_user_agent_string_here
GROQ_API_KEY=your_groq_api_key_here
""")

    print("Project setup complete!")
    print("Environment variables loaded.")
    print("Please create sample images in the 'images' folder using the naming convention 'location_name_description.jpg'")
    print("For example: 'shimla_mall_road.jpg', 'manali_solang_valley.jpg', etc.")

if __name__ == "__main__":
    setup_project()
