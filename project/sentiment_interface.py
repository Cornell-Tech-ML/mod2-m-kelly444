import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Load the pre-trained model and tokenizer
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

def render_run_sentiment_interface():
    """Render the sentiment analysis interface."""
    st.header("Sentiment Analysis")

    # Load the sentiment model
    sentiment_pipeline = load_model()

    # User input
    user_input = st.text_area("Enter text for sentiment analysis:", "I love this product!")

    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            results = sentiment_pipeline(user_input)
            sentiment_label = results[0]['label']
            sentiment_score = results[0]['score']
            st.success(f"Sentiment: **{sentiment_label}** (Score: {sentiment_score:.2f})")

    # Display examples of sentiment analysis
    if st.checkbox("Show Example Sentences"):
        st.subheader("Example Sentences")
        examples = [
            ("I love this movie!", "Positive"),
            ("This is the worst experience I've ever had.", "Negative"),
            ("It's okay, not great but not bad either.", "Neutral"),
            ("Absolutely fantastic! Highly recommend.", "Positive"),
            ("I don't like this product.", "Negative")
        ]
        for example, sentiment in examples:
            st.write(f"**Text:** {example} \n**Sentiment:** {sentiment}")

# Make this function callable from the main app
if __name__ == "__main__":
    render_run_sentiment_interface()
