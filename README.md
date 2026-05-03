# 🎓 College Support AI Chatbot

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/Gemini-API-orange.svg)](https://deepmind.google/technologies/gemini/)

## 📌 Project Overview

This project implements an **intelligent conversational chatbot** that answers college-related queries by extracting information from PDF documents. It demonstrates key AI concepts including:

- **Retrieval Augmented Generation (RAG)**
- **Natural Language Processing (NLP)**
- **Vector Similarity Search**
- **Prompt Engineering**

## 🚀 Features

- 📄 **PDF Processing**: Automatically reads and chunks college brochure PDFs
- 🔍 **Smart Retrieval**: Finds most relevant information using similarity search
- 🤖 **AI-Powered Responses**: Generates accurate answers using Google Gemini API
- 💬 **Chat Interface**: User-friendly web interface built with Streamlit
- 📊 **Source Attribution**: Shows which parts of documents were used

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.9+ | Core programming |
| Google Gemini API | LLM for response generation |
| Streamlit | Web interface |
| FAISS/TF-IDF | Vector similarity search |
| PyPDF2 | PDF text extraction |

## 📁 Project Structure
smart-chatbot/
├── data/
│ └── pdfs/
├── chatbot.py
├── requirements.txt
├── .env 
└── README.md
