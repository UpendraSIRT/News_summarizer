import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load API key
load_dotenv()
if "GOOGLE_API_KEY" not in os.environ:
    st.error("❌ Please set your GOOGLE_API_KEY in a .env file.")
    st.stop()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt templates for English and Hindi
english_prompt = PromptTemplate(
    template="Summarize the following news article concisely:\n\n{article}\n\nSummary:",
    input_variables=["article"]
)

hindi_prompt = PromptTemplate(
    template="निम्नलिखित समाचार लेख को संक्षेप में हिंदी में सारांशित करें:\n\n{article}\n\nसारांश:",
    input_variables=["article"]
)

# Cached function to extract text from article
@st.cache_data
def extract_news(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        return content if len(content) > 100 else "Could not extract enough content."
    except Exception as e:
        return f"Error: {e}"

# Function to summarize based on selected language
@st.cache_data
def summarize_article(url, language):
    article = extract_news(url)
    if article.startswith("Error") or "Could not" in article:
        return article

    if language == "Hindi":
        chain = LLMChain(llm=llm, prompt=hindi_prompt)
    else:
        chain = LLMChain(llm=llm, prompt=english_prompt)

    return chain.run(article=article)

# Streamlit UI
def main():
    st.set_page_config(page_title="News Article Summarizer", layout="centered")

    st.markdown("## 🧠✨ News Article Summarizer")
    st.markdown("Paste a news article URL below and get a concise summary!")

    st.markdown("#### 🌐 Select Language for Summary:")
    language = st.selectbox("Choose Language", ["English", "Hindi"])

    st.markdown("#### 🔗 Enter News Article URL:")
    url = st.text_input("Example: https://www.bbc.com/news/...", label_visibility="collapsed")

    if st.button("📄 Generate Summary", use_container_width=True, type="secondary"):
        if url:
            with st.spinner("⏳ Summarizing..."):
                summary = summarize_article(url, language)
                st.success("✅ Summary ready!")
                st.markdown("### 📝 Summary:")
                st.write(summary)
        else:
            st.warning("⚠️ Please enter a valid news article URL.")

if __name__ == "__main__":
    main()
