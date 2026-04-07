import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Spam Detection",
    page_icon="📩",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
    body {
        background-color: #f5f7fb;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }
    h1 {
        color: #4b4b4b;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
    }
    label {
        font-size: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Data --------------------
data = pd.read_csv(r"Spam-Detection/spam.csv")
data.drop_duplicates(inplace=True)

data['Category'] = data['Category'].replace({
    'ham': 'Not Spam',
    'spam': 'Spam'
})

X = data['Message']
y = data['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- Vectorization & Model --------------------
cv = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X_train_vec = cv.fit_transform(X_train)

model = MultinomialNB(alpha=0.5)
model.fit(X_train_vec, y_train)

# -------------------- Prediction Function --------------------
def predict_message(msg):
    msg_vec = cv.transform([msg])
    return model.predict(msg_vec)[0]

# -------------------- UI --------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("📩 Spam Detection App")

user_input = st.text_area(
    "Enter your message below:",
    height=120,
    placeholder="Type or paste your message here..."
)

if st.button("🔍 Check Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        prediction = predict_message(user_input)

        if prediction == "Spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT SPAM**")

st.markdown("</div>", unsafe_allow_html=True)

