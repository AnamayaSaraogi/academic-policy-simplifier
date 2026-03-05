import streamlit as st
import re
from PyPDF2 import PdfReader
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import spacy
import numpy as np
import networkx as nx
from difflib import SequenceMatcher

# ---------------- LOAD MODELS ---------------- #

nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Academic Policy Simplifier",
    layout="wide"
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Academic Rule Interpreter")

uploaded_file = st.sidebar.file_uploader(
    "Upload Policy (PDF or TXT)",
    type=["pdf", "txt"]
)

policy_text_input = st.sidebar.text_area(
    "Paste Policy Text",
    height=150
)

st.sidebar.markdown("---")
st.sidebar.caption("Upload a policy to begin analysis")

# side‑panel navigation shortcut
st.sidebar.markdown("[System Architecture](#system-architecture)")

# ---------------- FUNCTIONS ---------------- #

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    # Fix encoding issues
    text = text.encode("utf-8", "ignore").decode("utf-8")

    # Remove non-ASCII garbage
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Clean spacing
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def preprocess_text(text):
    text = text.lower()
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z0-9.%\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_sentences(text):
    sentences = sent_tokenize(text)

    cleaned = []
    for s in sentences:
        s = s.strip()

        # Skip very long lines (usually tables)
        if len(s) > 300:
            continue

        # Skip lines with too many numbers (table rows)
        if len(re.findall(r'\d+', s)) > 6:
            continue

        cleaned.append(s)

    return cleaned


def tokenize_and_remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words and len(w) > 2]


def generate_summary(sentences, words, max_sentences=5):
    """Generate a coherent summary of key sentences.
    
    Filters out fragments and grammatically unsound sentences, then
    ranks remaining sentences by word frequency + sentence importance.
    Returns up to max_sentences in their original document order.
    """
    
    if not sentences:
        return []
    
    # filter: keep only sentences with reasonable length and grammar
    # (avoid fragments and malformed extractions from PDFs)
    quality_sentences = []
    original_indices = []
    
    for idx, s in enumerate(sentences):
        s_clean = s.strip()
        
        # skip very short sentences (likely fragments)
        if len(s_clean) < 20:
            continue
        
        # skip sentences without a proper ending
        if not s_clean[-1] in '.!?':
            continue
        
        # skip sentences that look malformed (too many weird chars)
        alpha_ratio = sum(1 for c in s_clean if c.isalpha()) / len(s_clean)
        if alpha_ratio < 0.7:
            continue
        
        quality_sentences.append(s_clean)
        original_indices.append(idx)
    
    if not quality_sentences:
        return []
    
    # if few sentences remain, return them all
    if len(quality_sentences) <= max_sentences:
        return quality_sentences
    
    # score sentences by their content (word frequency + novelty)
    freq = Counter(words)
    scores = {}
    
    for sent in quality_sentences:
        sent_words = word_tokenize(sent.lower())
        # score = sum of word frequencies in this sentence
        score = sum(freq.get(w, 0) for w in sent_words)
        scores[sent] = score
    
    # pick top sentences by score and return in original order
    ranked_sents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    chosen_sents = [s for s, _ in ranked_sents[:max_sentences]]
    
    # restore original order for readability
    sent_order = {s: original_indices[quality_sentences.index(s)] 
                  for s in chosen_sents if s in quality_sentences}
    result = sorted(chosen_sents, key=lambda s: sent_order.get(s, float('inf')))
    
    return result


def extract_rules(sentences):
    rules = {
        "attendance": None,
        "condonation": None,
        "penalty": None,
        "deadlines": [],
        "eligibility": []
    }

    for sentence in sentences:
        s = sentence.lower()
        percent = re.findall(r'\d+%', s)

        if "attendance" in s and percent:
            rules["attendance"] = percent[0]

        if ("condone" in s or "shortage" in s) and percent:
            rules["condonation"] = percent[0]

        if "penalty" in s or "fine" in s or "deduct" in s:
            rules["penalty"] = sentence

        if "day" in s or "deadline" in s:
            rules["deadlines"].append(sentence)

        if "eligible" in s or "eligibility" in s:
            rules["eligibility"].append(sentence)

    return rules

def generate_suggested_questions(rules, sentences):
    """
    Generate intelligent suggested questions based on 
    detected rules and actual policy content.
    """
    suggestions = []

    if rules.get("attendance"):
        suggestions.append("What is the minimum attendance requirement?")

    if rules.get("condonation"):
        suggestions.append("Is condonation allowed for attendance shortage?")

    if rules.get("penalty"):
        suggestions.append("What are the penalties or fines mentioned?")

    if rules.get("deadlines"):
        suggestions.append("What are the important deadlines?")

    if rules.get("eligibility"):
        suggestions.append("What are the eligibility criteria?")

    # Extra intelligent detection using keywords
    full_text = " ".join(sentences).lower()

    if "exam" in full_text:
        suggestions.append("What are the exam-related rules?")

    if "assignment" in full_text:
        suggestions.append("Are there assignment submission rules?")

    if "fee" in full_text:
        suggestions.append("Are there any fee-related rules?")

    if "disciplinary" in full_text:
        suggestions.append("Are there disciplinary actions mentioned?")

    return suggestions


def analyze_sentiment(text):
    """
    Analyze sentiment and tone of the policy text.
    
    Returns:
    - polarity: -1 (negative) to 1 (positive)
    - subjectivity: 0 (objective) to 1 (subjective)
    - strictness: assessment of how strict the policy is
    - tone: formal, neutral, or positive
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    compound = scores['compound']
    
    # Determine strictness based on negative sentiment and certain keywords
    strict_keywords = ['must', 'shall', 'required', 'mandatory', 'prohibited', 'forbidden', 'penalty', 'fine', 'deduct']
    strict_count = sum(1 for keyword in strict_keywords if keyword in text.lower())
    strictness_level = min(100, (strict_count / max(len(text.split()), 1)) * 10000)
    
    # Determine tone
    if compound < -0.1:
        tone = "Strict/Formal (Negative)"
    elif compound < 0.1:
        tone = "Neutral/Formal"
    else:
        tone = "Positive/Constructive"
    
    return {
        'polarity': compound,
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu'],
        'strictness': strictness_level,
        'tone': tone
    }


def calculate_complexity_score(text, sentences):
    """
    Calculate reading complexity metrics of the policy.
    
    Metrics:
    - Flesch Reading Ease Score (0-100, higher = easier)
    - Average sentence length
    - Average word length
    - Overall complexity level
    """
    if not text or not sentences:
        return None
    
    # Count syllables (approximation)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiou"
        previous_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                count += 1
            previous_was_vowel = is_vowel
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        # Words have at least one syllable
        if count == 0:
            count = 1
        return count
    
    words = word_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Calculate metrics
    avg_sentence_length = word_count / max(sentence_count, 1)
    avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
    
    # Calculate total syllables
    total_syllables = sum(count_syllables(word) for word in words if word.isalpha())
    
    # Flesch Reading Ease Formula
    # Score = 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
    if sentence_count > 0 and word_count > 0:
        reading_ease = 206.835 - (1.015 * (word_count / sentence_count)) - (84.6 * (total_syllables / word_count))
        reading_ease = max(0, min(100, reading_ease))  # Clamp between 0-100
    else:
        reading_ease = 0
    
    # Determine reading level
    if reading_ease >= 90:
        level = "5th Grade"
    elif reading_ease >= 80:
        level = "6th Grade"
    elif reading_ease >= 70:
        level = "7th Grade"
    elif reading_ease >= 60:
        level = "8th-9th Grade"
    elif reading_ease >= 50:
        level = "10th-12th Grade"
    elif reading_ease >= 30:
        level = "College Level"
    else:
        level = "Graduate Level"
    
    # Determine complexity
    if reading_ease >= 60:
        complexity = "Low"
    elif reading_ease >= 40:
        complexity = "Medium"
    else:
        complexity = "High"
    
    return {
        'flesch_score': round(reading_ease, 2),
        'reading_level': level,
        'avg_sentence_length': round(avg_sentence_length, 2),
        'avg_word_length': round(avg_word_length, 2),
        'complexity': complexity,
        'word_count': word_count,
        'sentence_count': sentence_count
    }


def answer_question(question, sentences):
    """Find relevant sentences using semantic matching + fuzzy keyword matching.
    
    Handles spelling mistakes, word variations, and paraphrasing by combining:
    - Semantic similarity via spaCy vectors
    - Fuzzy keyword matching for spelling tolerance
    - Stemming for word variations (attend, attendance, attending, etc.)
    Returns all matching sentences sorted by relevance score.
    """
    if not question or not sentences:
        return []

    question_doc = nlp(question)
    if not question_doc.has_vector:
        return []

    # Extract key terms with both lemmatization and stemming
    stemmer = SnowballStemmer('english')
    question_tokens = [token for token in question_doc 
                      if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    
    # Create multiple representations of question terms for better matching
    question_lemmas = {token.lemma_.lower() for token in question_tokens}
    question_stems = {stemmer.stem(token.text.lower()) for token in question_tokens}
    question_raw = {token.text.lower() for token in question_tokens}
    
    scored_sentences = []

    for idx, sent_text in enumerate(sentences):
        sent_doc = nlp(sent_text)
        
        if not sent_doc.has_vector:
            continue

        # 1. Semantic similarity (captures meaning/paraphrasing)
        semantic_sim = question_doc.similarity(sent_doc)

        # 2. Extract sentences terms for comparison
        sent_tokens = [token for token in sent_doc 
                      if token.pos_ in ['NOUN', 'VERB', 'ADJ']]
        sent_lemmas = {token.lemma_.lower() for token in sent_tokens}
        sent_stems = {stemmer.stem(token.text.lower()) for token in sent_tokens}
        sent_raw = {token.text.lower() for token in sent_tokens}
        
        # 3. Calculate keyword overlap (exact match on lemmas)
        exact_overlap = len(question_lemmas & sent_lemmas)
        
        # 4. Calculate stem-based overlap (handles word variations)
        stem_overlap = len(question_stems & sent_stems)
        
        # 5. Fuzzy matching for spelling tolerance (handles typos)
        fuzzy_matches = 0
        for q_term in question_raw:
            for s_term in sent_raw:
                # Check if terms are similar enough (handling spelling mistakes)
                similarity = SequenceMatcher(None, q_term, s_term).ratio()
                if similarity >= 0.75:  # 75% match catches most typos
                    fuzzy_matches += 1
                    break  # Count each question term only once
        
        # 6. Combined scoring
        # Give more weight to exact matches, then stem matches, then fuzzy
        keyword_score = (exact_overlap * 0.5 + stem_overlap * 0.3 + fuzzy_matches * 0.2) / max(len(question_lemmas), 1)
        
        # Combine semantic similarity with keyword matching
        # Lowered thresholds to catch more relevant results
        combined_score = (semantic_sim * 0.6) + (keyword_score * 0.4)
        
        # Keep sentence if it passes minimum thresholds
        # Lowered threshold from 0.35 to 0.25 for better recall
        if combined_score >= 0.25 or keyword_score >= 0.3:
            scored_sentences.append((sent_text, combined_score, keyword_score))

    # Sort by combined score (descending) and return all matches
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    return [sent for sent, _, _ in scored_sentences]


def highlight_entities(text):
    doc = nlp(text)

    colors = {
        "ORG": "#ffd54f",
        "PERSON": "#81d4fa",
        "GPE": "#a5d6a7",
        "DATE": "#ffab91",
        "LAW": "#ce93d8",
        "NORP": "#f48fb1",
        "CARDINAL": "#b0bec5"
    }

    highlighted = text

    for ent in reversed(doc.ents):
        color = colors.get(ent.label_, "#e0e0e0")
        span = f'<span style="background-color:{color}; padding:2px 4px; border-radius:4px;">{ent.text}</span>'
        highlighted = highlighted[:ent.start_char] + span + highlighted[ent.end_char:]

    return highlighted


# ---------------- LOAD POLICY ---------------- #

policy_text = ""

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        policy_text = extract_text_from_pdf(uploaded_file)
    else:
        policy_text = uploaded_file.read().decode("utf-8")
elif policy_text_input.strip():
    policy_text = policy_text_input


# ---------------- NLP PROCESSING ---------------- #

processed_text = ""
sentences = []
filtered_words = []
summary_sentences = []
rules = {}

if policy_text:
    processed_text = preprocess_text(policy_text)
    sentences = get_sentences(policy_text)
    filtered_words = tokenize_and_remove_stopwords(processed_text)
    summary_sentences = generate_summary(sentences, filtered_words, max_sentences=5)
    rules = extract_rules(sentences)
    suggested_questions = generate_suggested_questions(rules, sentences)

st.title("Academic Policy Simplifier")
st.caption("Understand institutional policies instantly")

# ---------------- TABS ---------------- #

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Q&A Assistant",
    "Analytics",
    "Architecture"
])

# ---------- TAB 1: OVERVIEW ---------- #

with tab1:
    if policy_text:
        st.success("Policy successfully analyzed.")

        st.subheader("Summary")

        if summary_sentences:
            for s in summary_sentences:
                st.markdown(f"- {s}")
        else:
            st.write("No summary could be generated.")

        # also surface the key rule values so that users immediately see
        # the most important policy elements without hunting through
        # the document
        if rules:
            st.markdown("**Key rules extracted:**")
            if rules.get("attendance"):
                st.markdown(f"- Attendance required: {rules['attendance']}")
            if rules.get("condonation"):
                st.markdown(f"- Condonation limit: {rules['condonation']}")
            if rules.get("penalty"):
                st.markdown(f"- Penalty/fine text: {rules['penalty']}")
            if rules.get("deadlines"):
                for d in rules["deadlines"]:
                    st.markdown(f"- Deadline sentence: {d}")
            if rules.get("eligibility"):
                for e in rules["eligibility"]:
                    st.markdown(f"- Eligibility sentence: {e}")

        st.subheader("Quick Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Attendance Required", rules.get("attendance", "—"))
        col2.metric("Condonation Limit", rules.get("condonation", "—"))
        col3.metric("Penalty Rule", "Detected" if rules.get("penalty") else "—")

    else:
        st.info("Upload or paste a policy to begin.")

# ---------- TAB 2: Q&A ---------- #

with tab2:
    st.subheader("Ask Questions About the Policy")

    if policy_text:
        # Initialize session state for question
        if 'qa_question' not in st.session_state:
            st.session_state.qa_question = ""

        # ---------------- Suggested Questions ---------------- #
        if suggested_questions:
            st.markdown("### Suggested Questions")
            st.markdown("Click any question below to select it:")
            
            # Create a more visually appealing suggested questions section
            cols = st.columns(2)
            for i, question in enumerate(suggested_questions):
                col = cols[i % 2]
                if col.button(
                    question, 
                    key=f"suggest_{i}",
                    use_container_width=True,
                ):
                    st.session_state.qa_question = question

            st.markdown("---")

        # Question input with session state
        user_query = st.text_input(
            "Your Question:",
            value=st.session_state.qa_question,
            placeholder="Ask anything about the policy...",
            help="Type your question or click a suggested question above"
        )
        
        # Update session state when user types
        st.session_state.qa_question = user_query

        if user_query:
            answers = answer_question(user_query, sentences)

            if answers:
                st.success(f"Found {len(answers)} relevant result(s):")
                st.markdown("---")
                for i, answer in enumerate(answers, 1):
                    with st.container():
                        st.markdown(f"**Result {i}:**")
                        st.markdown(f"> {answer}")
                        st.markdown("")
            else:
                st.warning("No relevant rule found in the policy. Try rephrasing your question.")

    else:
        st.info("Load a policy first to ask questions.")



# ---------- TAB 3: ANALYTICS ---------- #

with tab3:
    st.subheader("Policy Analytics")

    if policy_text:
        st.markdown("### Sentiment & Tone Analysis")
        
        # Calculate sentiment
        sentiment_data = analyze_sentiment(policy_text)
        
        # Display sentiment metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Policy Tone", sentiment_data['tone'])
        
        with col2:
            polarity_pct = round((sentiment_data['polarity'] + 1) / 2 * 100, 1)
            st.metric("Polarity Score", f"{polarity_pct}%")
        
        with col3:
            st.metric("Strictness Level", f"{round(sentiment_data['strictness'], 1)}%")
        
        # Sentiment breakdown
        st.markdown("**Sentiment Breakdown:**")
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.metric("Positive", f"{round(sentiment_data['positive'] * 100, 1)}%")
        
        with sentiment_col2:
            st.metric("Neutral", f"{round(sentiment_data['neutral'] * 100, 1)}%")
        
        with sentiment_col3:
            st.metric("Negative", f"{round(sentiment_data['negative'] * 100, 1)}%")
        
        # Create sentiment visualization
        sentiment_values = [sentiment_data['positive'], sentiment_data['neutral'], sentiment_data['negative']]
        sentiment_labels = ['Positive', 'Neutral', 'Negative']
        colors = ['#2ecc71', '#95a5a6', '#e74c3c']
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(sentiment_labels, sentiment_values, color=colors)
        ax.set_xlabel('Proportion')
        ax.set_title('Policy Sentiment Distribution')
        st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### Complexity & Readability Analysis")
        
        # Calculate complexity
        complexity_data = calculate_complexity_score(policy_text, sentences)
        
        if complexity_data:
            # Display complexity metrics in a table
            complexity_metrics = {
                "Metric": [
                    "Reading Level",
                    "Flesch Reading Ease",
                    "Avg Sentence Length",
                    "Avg Word Length",
                    "Overall Complexity",
                    "Total Words",
                    "Total Sentences"
                ],
                "Score": [
                    complexity_data['reading_level'],
                    f"{complexity_data['flesch_score']}/100",
                    f"{complexity_data['avg_sentence_length']} words",
                    f"{complexity_data['avg_word_length']} chars",
                    complexity_data['complexity'],
                    f"{complexity_data['word_count']} words",
                    f"{complexity_data['sentence_count']} sentences"
                ]
            }
            
            # Display as styled table
            st.markdown("**Readability Metrics:**")
            st.dataframe(
                complexity_metrics,
                use_container_width=True,
                hide_index=True
            )
            
            # Complexity visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Flesch score visualization
            flesch = complexity_data['flesch_score']
            colors_flesch = ['#e74c3c' if flesch < 30 else '#f39c12' if flesch < 60 else '#2ecc71']
            ax1.barh(['Readability'], [flesch], color=colors_flesch)
            ax1.set_xlim(0, 100)
            ax1.set_xlabel('Flesch Reading Ease Score')
            ax1.set_title('Reading Ease (Higher = Easier)')
            ax1.text(flesch/2, 0, f'{flesch:.1f}', va='center', ha='center', color='white', fontweight='bold')
            
            # Average metrics
            metrics_names = ['Sentence\nLength', 'Word\nLength']
            metrics_values = [
                complexity_data['avg_sentence_length'],
                complexity_data['avg_word_length']
            ]
            ax2.bar(metrics_names, metrics_values, color=['#3498db', '#9b59b6'])
            ax2.set_ylabel('Length')
            ax2.set_title('Average Length Metrics')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        st.markdown("### Interpretation Guide")
        st.markdown("""
        **Reading Level Legend:**
        - 5th-6th Grade: Very easy to read, simple language
        - 7th-9th Grade: Moderately simple, general audience
        - 10th-12th Grade: Standard difficulty level
        - College Level: Requires college education
        - Graduate Level: Complex vocabulary and structure
        
        **Complexity Levels:**
        - **Low**: Easy to understand, accessible language
        - **Medium**: Moderately complex, requires attention
        - **High**: Complex structure, legal/technical terminology
        
        **Policy Tone:**
        - **Strict/Formal**: Policy contains mandatory requirements and penalties
        - **Neutral/Formal**: Objective tone, clear rules
        - **Positive/Constructive**: Encouraging and supportive language
        """)

    else:
        st.info("Load a policy first to view analytics.")



# ---------- TAB 4: ARCHITECTURE ---------- #

with tab4:
    # anchor for sidebar link
    st.markdown('<a name="system-architecture"></a>', unsafe_allow_html=True)
    st.subheader("System Architecture")
    st.markdown(
        """
        The application processes policies through several stages. Below
        we also expose the intermediate outputs from each preprocessing step
        so you can see exactly how the text transforms.
        """,
        unsafe_allow_html=True,
    )

    if policy_text:
        st.markdown("---")
        st.markdown("### 1. Raw Text Extraction")
        st.write(policy_text[:1000] + ("..." if len(policy_text) > 1000 else ""))

        st.markdown("### 2. Cleaning / Normalization")
        st.write(processed_text[:1000] + ("..." if len(processed_text) > 1000 else ""))

        st.markdown("### 3. Sentence Splitting")
        with st.expander("View all extracted sentences"):
            for s in sentences[:20]:
                st.write(f"- {s}")
            if len(sentences) > 20:
                st.write(f"...and {len(sentences)-20} more sentences")

        st.markdown("### 4. Tokenization & Stopword Removal")
        with st.expander("Top 50 filtered words"):
            st.write(filtered_words[:50])

        st.markdown("### 5. Rule Extraction Example")
        st.write(rules)

    else:
        st.info("Load a policy to see architecture details and intermediate outputs.")

    # display cleaned text example
    if policy_text:
        st.markdown("### Cleaned Text Preview")
        with st.expander("View the full cleaned/normalized output"):
            st.write(processed_text)


    # display architecture diagram from local file (resized)
    st.markdown("### Visual Overview of the Pipeline")
    st.markdown("Below is a graphical representation of the system components and the flow of data between them. "
                "Each box corresponds to a major processing stage described above.")
    st.image("architecture.png", caption="System architecture flow",
             width=400)  # specify a reasonable width