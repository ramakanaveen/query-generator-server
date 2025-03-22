# test_llm_connections.py
import os
from pathlib import Path
from dotenv import load_dotenv
import time

def test_vertex_connection():
    """
    Test connection to Google Vertex AI using the provided credentials.
    """
    try:
        # Get environment variables
        credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
        project_id = os.getenv("GOOGLE_PROJECT_ID")
        location = os.getenv("GOOGLE_LOCATION", "us-central1")
        
        # Check if credentials path is provided
        if not credentials_path:
            print("ERROR: GOOGLE_CREDENTIALS_PATH environment variable not set.")
            return False
            
        # Check if project ID is provided
        if not project_id:
            print("ERROR: GOOGLE_PROJECT_ID environment variable not set.")
            return False
            
        # Check if credentials file exists
        credentials_file = Path(credentials_path)
        if not credentials_file.exists():
            print(f"ERROR: Credentials file not found at {credentials_path}")
            return False
            
        print(f"Using credentials file: {credentials_path}")
        print(f"Using project ID: {project_id}")
        print(f"Using location: {location}")
        
        # Set the environment variable for Google authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)
        
        # Use delayed import to avoid dependencies if only testing Claude
        from google.cloud import aiplatform
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        # Initialize Vertex AI
        print("Initializing Vertex AI...")
        vertexai.init(project=project_id, location=location)
        
        # Try accessing a model to verify permissions
        print("Attempting to access Gemini model...")
        model = GenerativeModel("gemini-pro")
        
        # Generate a simple response as a test
        print("Testing model with a simple prompt...")
        response = model.generate_content("Hello, are you working correctly?")
        
        print(f"\nGemini response: {response.text}\n")
        print("✅ Connection to Google Vertex AI successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Google Vertex AI: {str(e)}")
        return False

def test_claude_connection():
    """
    Test connection to Anthropic's Claude API using the provided API key.
    """
    try:
        # Get API key
        api_key = os.getenv("API_KEY_ANTHROPIC")
        
        # Check if API key is provided
        if not api_key:
            print("ERROR: API_KEY_ANTHROPIC environment variable not set.")
            return False
            
        print("Using Anthropic API key:", api_key[:4] + "..." + api_key[-4:])
        
        # Use delayed import to avoid dependencies if only testing Vertex
        from anthropic import Anthropic
        
        # Initialize Anthropic client
        print("Initializing Anthropic client...")
        client = Anthropic(api_key=api_key)
        
        # Generate a simple response as a test
        print("Testing Claude with a simple prompt...")
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            temperature=0.2,
            messages=[
                {"role": "user", "content": "Hello, are you working correctly?"}
            ]
        )
        
        print(f"\nClaude response: {message.content[0].text}\n")
        print("✅ Connection to Anthropic Claude successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Anthropic Claude: {str(e)}")
        return False

if __name__ == "__main__":
    load_dotenv()
    
    print("=" * 50)
    print("TESTING LLM CONNECTIONS")
    print("=" * 50)
    
    print("\n[1/2] Testing Google Vertex AI Connection...")
    vertex_result = test_vertex_connection()
    
    print("\n" + "=" * 50)
    
    print("\n[2/2] Testing Anthropic Claude Connection...")
    claude_result = test_claude_connection()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Google Vertex AI: {'✅ PASSED' if vertex_result else '❌ FAILED'}")
    print(f"Anthropic Claude: {'✅ PASSED' if claude_result else '❌ FAILED'}")
    print("=" * 50)
    
    if vertex_result and claude_result:
        print("\n🎉 All LLM connections are working properly!")
    else:
        print("\n⚠️ Some LLM connections failed. Please check the error messages above.")