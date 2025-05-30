from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests
from bs4 import BeautifulSoup
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st

# Load environment variables (ensure .env file is in the same directory or adjust path)
load_dotenv()

# --- Initialize Gemini Model ---
# Ensure GOOGLE_API_KEY is set in your .env file
# It's good practice to check if the API key is available
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")
    st.stop() # Stop the app if API key is missing

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# --- Functions for News Extraction and Summarization ---

@st.cache_data(show_spinner=False) # Cache the results of this function
def extract_news(url):
    """
    Extracts text content from a given news article URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        
        # Basic check for minimal content
        if len(text) < 100: # If the extracted text is too short, it might be an issue
            return "Failed to extract substantial content from the URL. The page might be empty or structured differently."
        return text
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch news from {url}: {e}"
    except Exception as e:
        return f"An unexpected error occurred while processing {url}: {e}"

# Prompt template for summarization
summarize_prompt = PromptTemplate(
    template="Summarize the following news article concisely, highlighting the main points:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

# Create the LLMChain
summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

@st.cache_data(show_spinner=False) # Cache the results of this function
def summarize_news_article(url):
    """
    Summarizes a news article from a given URL using the LLMChain.
    """
    st.info(f"Attempting to fetch and summarize: {url}")
    article_content = extract_news(url)

    if article_content.startswith("Failed to fetch") or article_content.startswith("Failed to extract"):
        return article_content # Return the error message directly

    try:
        # Show a spinner while the LLM is working
        with st.spinner("Generating summary... This might take a moment."):
            summary = summarize_chain.run(article=article_content)
        return summary
    except Exception as e:
        return f"An error occurred during summarization: {e}"

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="News Article Summarizer", page_icon="📰")

    st.markdown(
        """
        <style>
        .main-header {
            font-size: 3em;
            color: #4CAF50; /* Green */
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial Black', sans-serif;
        }
        .subheader {
            font-size: 1.5em;
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .summary-box {
            background-color: #e6ffe6; /* Light green background */
            border-left: 5px solid #4CAF50; /* Green border */
            padding: 20px;
            margin-top: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="main-header">📰 News Article Summarizer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Paste a news article URL below to get a concise summary!</p>', unsafe_allow_html=True)

    user_url = st.text_input(
        "Enter the news article URL:",
        placeholder="e.g., https://www.nytimes.com/..."
    )

    if st.button("Summarize Article"):
        if user_url:
            with st.spinner("Fetching and summarizing..."):
                summary_result = summarize_news_article(user_url)

            if summary_result.startswith(("Failed to fetch", "Failed to extract", "An error occurred")):
                st.error(summary_result)
            else:
                st.markdown(f"---")
                st.subheader("Summary:")
                st.markdown(f'<div class="summary-box">{summary_result}</div>', unsafe_allow_html=True)
                st.markdown(f"---")
                st.info(f"Original Article URL: {user_url}")
        else:
            st.warning("Please enter a URL to summarize.")

    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses Google's Gemini-1.5-Flash model to summarize news articles. "
        "Simply paste a URL, and let the AI do the work!"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**How it works:**")
    st.sidebar.markdown(
        "- Fetches content from the provided URL."
        "- Extracts relevant text using BeautifulSoup."
        "- Sends the text to the Gemini model for summarization."
        "- Displays the concise summary."
    )


if __name__ == "__main__":
    main()