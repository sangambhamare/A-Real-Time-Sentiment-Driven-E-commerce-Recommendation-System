import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# Cache the summarization pipeline to avoid reloading it on every run.
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def fetch_reviews(url):
    """
    Fetch reviews from the given product URL.
    Adjust the HTML selectors based on the page structure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/90.0.4430.93 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching page: {e}")
        return ""
    
    soup = BeautifulSoup(response.content, "html.parser")
    review_elements = soup.find_all("span", {"data-hook": "review-body"})
    reviews = [elem.get_text(strip=True) for elem in review_elements]
    return " ".join(reviews)

def chunk_text(text, max_words=500):
    """
    Splits text into smaller chunks with a maximum number of words.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def summarize_text(text):
    """
    Use the Hugging Face summarization model to summarize the provided text.
    If the text is too long, it splits it into chunks and summarizes each chunk.
    """
    if not text.strip():
        return "No content to summarize."
    
    words = text.split()
    # Split into chunks if necessary
    if len(words) > 500:
        chunks = chunk_text(text, max_words=500)
    else:
        chunks = [text]
    
    summaries = []
    for chunk in chunks:
        try:
            summary_output = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            if summary_output and isinstance(summary_output, list):
                summaries.append(summary_output[0]['summary_text'])
        except Exception as e:
            summaries.append(f"Error during summarization: {e}")
    overall_summary = " ".join(summaries)
    return overall_summary

def main():
    st.title("SmartShop Review Summarizer")
    st.write("A real-time, sentiment-driven review summarizer for e-commerce product pages.")
    
    product_url = st.text_input("Enter the product URL:")
    
    if st.button("Get Review Summary"):
        if not product_url:
            st.error("Please enter a valid URL.")
        else:
            with st.spinner("Fetching reviews..."):
                reviews_text = fetch_reviews(product_url)
            if not reviews_text:
                st.error("No reviews found or an error occurred.")
            else:
                with st.spinner("Summarizing reviews..."):
                    summary = summarize_text(reviews_text)
                st.subheader("Review Summary:")
                st.write(summary)

if __name__ == "__main__":
    main()
