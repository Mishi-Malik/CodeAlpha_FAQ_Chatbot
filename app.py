from faqs_data import FAQS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# --- NLTK DOWNLOADS ---
try:
  nltk.data.find("tokenizers/punkt_tab")
  nltk.data.find("corpora/stopwords")
  nltk.data.find("corpora/wordnet")
except LookupError:
  nltk.download("punkt_tab")
  nltk.download("stopwords")
  nltk.download("wordnet")

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FAQ.AI | Smart Assistant", page_icon="✦", layout="wide"
)

# --- LOAD CSS FILE ---
try:
  with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except:
  pass

# --- SIDEBAR NAVIGATION (Cleaned up) ---
with st.sidebar:
  st.markdown(
      "<h2 style='color: white; font-size: 1.3rem; margin-bottom: 0;'>🤖"
      " FAQ.AI</h2>",
      unsafe_allow_html=True,
  )
  st.markdown(
      "<p style='color: #94a3b8; font-size: 0.75rem; margin-top: 2px;"
      " margin-bottom: 20px;'>Smart Assistant Portal</p>",
      unsafe_allow_html=True,
  )

  nav_option = st.radio(
      "Navigation",
      [
          "💬  Chat Assistant",
          "📚  Knowledge Base",
          "📂  Categories",
          "⚙️  How It Works",
      ],
      label_visibility="collapsed",
  )

  st.markdown("<br><br><br>", unsafe_allow_html=True)
  st.markdown(
      """
        <div style='background: #1e293b; padding: 14px; border-radius: 12px; text-align: center; border: 1px solid #334155;'>
            <p style='margin:0; font-weight:600; font-size: 0.82rem; color: #fff;'>Need Help?</p>
            <p style='font-size:0.72rem; color:#94a3b8; margin:3px 0 0 0;'>CodeAlpha Internship Task</p>
        </div>
    """,
      unsafe_allow_html=True,
  )

# --- NLTK TEXT PREPROCESSING ---


def preprocess_text(text):
  if not isinstance(text, str):
    return ""
  tokens = word_tokenize(text.lower())
  lemmatizer = WordNetLemmatizer()
  stop_words = set(stopwords.words("english"))
  cleaned_tokens = [
      lemmatizer.lemmatize(word)
      for word in tokens
      if word.isalnum() and word not in stop_words
  ]
  return " ".join(cleaned_tokens)


# Extract questions and answers
questions = [item["question"] for item in FAQS]
answers = [item["answer"] for item in FAQS]

preprocessed_questions = [preprocess_text(q) for q in questions]
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(preprocessed_questions)

# --- SESSION STATE ---
if "messages" not in st.session_state:
  st.session_state.messages = [{
      "role": "assistant",
      "content": (
          "Hello! I am your CodeAlpha AI Assistant. How can I help you with"
          " your internship tasks, duration, or guidelines today?"
      ),
  }]

if "feedback" not in st.session_state:
  st.session_state.feedback = {}

# --- MAIN DASHBOARD VIEW ---
if nav_option == "💬  Chat Assistant":
  st.markdown(
      """
        <div class="hero-banner">
            <h1 class="hero-title">💡 Smart AI FAQ Assistant</h1>
            <p class="hero-subtitle">CodeAlpha Internship Portal — Advanced NLP & Scikit-Learn Engine</p>
        </div>
    """,
      unsafe_allow_html=True,
  )

  for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
      st.markdown(message["content"], unsafe_allow_html=True)

      if message["role"] == "assistant" and idx > 0:
        fb_key = f"fb_{idx}"
        if fb_key not in st.session_state.feedback:
          cols = st.columns([1.2, 1.2, 7.6])
          with cols[0]:
            if st.button("👍 Helpful", key=f"help_{idx}"):
              st.session_state.feedback[fb_key] = "Thank you for your feedback!"
              st.rerun()
          with cols[1]:
            if st.button("👎 Poor", key=f"not_help_{idx}"):
              st.session_state.feedback[fb_key] = (
                  "Thanks! We will improve this answer."
              )
              st.rerun()
        else:
          st.success(f"✓ {st.session_state.feedback[fb_key]}")

  if user_query := st.chat_input("Ask a question about your internship..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
      st.markdown(user_query)

    with st.chat_message("assistant"):
      with st.spinner("Analyzing semantic patterns..."):
        processed_query = preprocess_text(user_query)
        user_vector = vectorizer.transform([processed_query])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_idx = similarities.argmax()
        best_score = similarities[0, best_match_idx]
        match_percentage = int(best_score * 100)

        if best_score > 0.05:
          bot_response = answers[best_match_idx]
          matched_q = questions[best_match_idx]
          formatted_response = (
              f"<div class='match-box'>🟢 Best Match Found"
              f" ({match_percentage}% confidence) —"
              f" <i>\"{matched_q}\"</i></div><div style='line-height: 1.6;'>"
              f" {bot_response}</div>"
          )
        else:
          formatted_response = (
              "I couldn't find a direct match in my knowledge base. Please try"
              " asking about internship duration, tasks, deadlines, or"
              " submissions!"
          )

        st.markdown(formatted_response, unsafe_allow_html=True)
        st.session_state.messages.append(
            {"role": "assistant", "content": formatted_response}
        )
        st.rerun()

elif nav_option == "📚  Knowledge Base":
  st.markdown("<h2>📚 Knowledge Base</h2>", unsafe_allow_html=True)
  st.markdown("---")
  for i, item in enumerate(FAQS, 1):
    with st.expander(f"Q{i}: {item['question']} ({item['category']})"):
      st.write(item["answer"])

elif nav_option == "📂  Categories":
  st.markdown("<h2>📂 FAQ Categories Overview</h2>", unsafe_allow_html=True)
  for cat in ["General", "Internship", "Support"]:
    count = sum(1 for item in FAQS if item["category"] == cat)
    st.info(f"**{cat}** Category — {count} Questions Available")

elif nav_option == "⚙️  How It Works":
  st.markdown("<h2>⚙️ How It Works (NLP Architecture)</h2>", unsafe_allow_html=True)
  st.markdown(
      "1. **Text Preprocessing**: Tokenization and Lemmatization using NLTK."
  )
  st.markdown(
      "2. **Vectorization**: TF-IDF Vectorizer converts text into numerical"
      " weights."
  )
  st.markdown(
      "3. **Similarity Matching**: Cosine similarity algorithm retrieves the"
      " accurate answer."
  )