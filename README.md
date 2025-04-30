# 🧠 Home Assignment 4 – CS5720: Neural Networks and Deep Learning  
**University of Central Missouri – Spring 2025**  
**Student Name:** Manikanta Rajulapati

**Course:** CS5720  


---

## 📌 Overview

This assignment includes implementations of key NLP and deep learning concepts:
- NLP text preprocessing (tokenization, stopword removal, stemming)
- Named Entity Recognition using SpaCy
- Scaled dot-product attention (core of Transformer models)
- Sentiment analysis using HuggingFace’s pre-trained models

Each question has been solved in Python using popular libraries such as **NLTK, SpaCy, NumPy, and HuggingFace Transformers**.

---

## ✅ Q1: NLP Preprocessing Pipeline

### 🔧 Task:
Write a function that performs three basic NLP preprocessing steps:
1. **Tokenization** – Splits a sentence into words and punctuation.
2. **Stopword Removal** – Removes common low-information words (e.g., *the*, *in*, *are*).
3. **Stemming** – Reduces words to their root form using PorterStemmer.

### 🧪 Input Sentence:
"NLP techniques are used in virtual assistants like Alexa and Siri."

### 📤 Expected Output:
- **Original Tokens**: All words and punctuation split from the sentence.
- **Tokens Without Stopwords**: Only meaningful words remain.
- **Stemmed Words**: Each word is reduced to its base/root form.

### 📘 Conceptual Notes:
- **Stemming vs Lemmatization**: Stemming chops off word endings crudely, while lemmatization returns the dictionary form.  
- **Stopword Removal Use Case**: Useful for classification and topic modeling, but might harm performance in sentiment analysis where words like “not” are critical.

---

## ✅ Q2: Named Entity Recognition (NER) with SpaCy

### 🔧 Task:
Use the `spaCy` NLP library to:
- Identify named entities from a sentence.
- Print each entity’s text, label (e.g., PERSON, DATE), and character span.

### 🧪 Input Sentence:
"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

### 📤 Expected Output:
Each named entity is printed with:
- **Text**: "Barack Obama"
- **Label**: PERSON, DATE, ORG, etc.
- **Start/End Char**: Position in the original sentence.

### 📘 Conceptual Notes:
- **NER vs POS Tagging**: NER identifies real-world entities; POS tagging marks grammatical roles like noun, verb, etc.
- **Real-World Applications**: Search engines, financial news extraction, legal document summarization, and more.

---

## ✅ Q3: Scaled Dot-Product Attention

### 🔧 Task:
Implement scaled dot-product attention — the mathematical foundation of the Transformer architecture.

### 📘 Steps:
1. Compute **Q · Kᵀ**
2. Scale by **√d** (dimension of the key vectors)
3. Apply **softmax** to get attention weights
4. Multiply by **V** (Value matrix) to get the output

### 🧪 Test Input:
```python
Q = [[1, 0, 1, 0], [0, 1, 0, 1]]
K = [[1, 0, 1, 0], [0, 1, 0, 1]]
V = [[1, 2, 3, 4], [5, 6, 7, 8]]
📤 Expected Output:
Attention Weights Matrix

Final Output Matrix (weighted sum of values)

📘 Conceptual Notes:
Why scale by √d?: Prevents large dot products from dominating softmax — improves numerical stability.

Use in Self-Attention: Helps the model focus on contextually relevant parts of the input.

## 📌 Overview

This project demonstrates the use of a **pre-trained sentiment analysis pipeline** from HuggingFace’s `transformers` library.  
Given an input sentence, the model predicts whether the sentiment is **positive** or **negative**, along with a **confidence score**.

---

## ✅ Task Description

### ✔️ Objective:
- Use HuggingFace’s `pipeline()` API to:
  - Load a pre-trained sentiment classifier
  - Analyze the sentiment of a provided sentence
  - Print the sentiment **label** and **confidence score**

### 🧪 Input Sentence: "Despite the high price, the performance of the new MacBook is outstanding."

### 📤 Expected Output:
Sentiment: POSITIVE
Confidence Score: 0.9987


# ✅ Q4: Sentiment Analysis using HuggingFace Transformers  
  ---

## 📌 Overview

This task demonstrates the use of a **pre-trained sentiment analysis pipeline** from HuggingFace’s `transformers` library.  
The goal is to classify the sentiment of a given sentence as **positive** or **negative**, and provide a **confidence score**.

---

## ✔️ Objective

The following steps are performed in this task:

- Load a **pre-trained sentiment classifier** using HuggingFace’s `pipeline()` API  
- Analyze the sentiment of a given English sentence  
- Print the **sentiment label** and **confidence score**

---

## 🧪 Input Sentence

```text
Despite the high price, the performance of the new MacBook is outstanding.

Output:
Sentiment: POSITIVE
Confidence Score: 0.9987
