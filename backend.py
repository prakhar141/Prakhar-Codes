import sys
print(sys.executable)
print(sys.version)

import os
try:
    from rapidfuzz import fuzz
except ImportError:
    os.system("pip install rapidfuzz==3.6.1")
    from rapidfuzz import fuzz

import streamlit as st
import re
import pickle
from collections import defaultdict
from functools import lru_cache
from rapidfuzz import fuzz

# ---------------- LOAD PRECOMPUTED FILES ----------------
@st.cache_resource
def load_resources():
    with open("q_table.pkl", "rb") as f:
        q_table = defaultdict(float, pickle.load(f))
    with open("correction_map.pkl", "rb") as f:
        correction_map = pickle.load(f)

    # Build vocab from correction_map + their candidates
    vocab = set()
    for word, candidates in correction_map.items():
        vocab.add(word)
        vocab.update(w for w, _ in candidates)
    return q_table, correction_map, sorted(vocab)

q_table, correction_map, vocab = load_resources()
vocab_set = set(vocab)

# ---------------- REGEX PATTERNS ----------------
word_pattern = re.compile(r'\b\w+\b|\W+')
token_pattern = re.compile(r'\w+')

@lru_cache(maxsize=10000)
def lev_ratio_cached(a, b):
    return fuzz.token_sort_ratio(a, b) / 100

# ---------------- CORRECT SENTENCE ----------------
def correct_sentence_advanced(input_sentence, sim_threshold=0.78):
    corrected = []
    highlighted = []
    words = word_pattern.findall(input_sentence.lower())

    for word in words:
        if not token_pattern.match(word):
            corrected.append(word)
            highlighted.append(word)
            continue

        if word in vocab_set:
            corrected.append(word)
            highlighted.append(word)
            continue

        candidates = correction_map.get(word, None)
        if not candidates:
            filtered = [v for v in vocab if abs(len(v) - len(word)) <= 2 and v[0] == word[0]]
            if not filtered:
                filtered = vocab
            candidates = [(v, lev_ratio_cached(word, v)) for v in filtered]

        best_word, best_score = max(
            candidates,
            key=lambda x: q_table.get((word, x[0]), x[1])
        )

        final_word = best_word if best_score >= sim_threshold else word
        corrected.append(final_word)

        if final_word != word:
            highlighted.append(f":red[`{word}`] → :green[`{final_word}`]")
        else:
            highlighted.append(final_word)

    return ''.join(corrected), ' '.join(highlighted)

# ---------------- STREAMLIT UI ----------------
st.title("🧠 Intelligent Spell Corrector")
st.markdown("Fixes spelling errors in your sentence using AI trained with Reinforcement Learning.")

input_text = st.text_area("✏️ Enter a sentence with spelling mistakes:", "Prakhar  maade me and I am redy to hlp you")

if st.button("🔧 Fix My Sentence"):
    with st.spinner("Analyzing and correcting..."):
        corrected_text, visual_feedback = correct_sentence_advanced(input_text)
        st.markdown("### ✅ Final Correction:")
        st.success(corrected_text)

        st.markdown("### ✨ What Changed?")
        st.markdown(visual_feedback)
