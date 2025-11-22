# 🌐 URL-Based RAG Q&A Assistant (Gemini 2.5 Flash)

A simple project that lets you enter one or more URLs, extract clean text from each page, store it in a vector database, and ask questions about the content using **Gemini 2.5 Flash**.  
This project uses **LangChain**, **FAISS**, and **Gradio** to build a smooth RAG workflow.


## 📸 Project Screenshots

### 🖼️ Screenshot 1 — Main Interface  
*(replace with your own image)*  
![Screenshot 1](images/screenshot1.png)

### 🖼️ Screenshot 2 — Answer Example  
*(replace with your own image)*  
![Screenshot 2](images/screenshot2.png)


## 🚀 Features

- Enter one or more URLs  
- Automatic extraction and cleaning of webpage text  
- Chunking and embedding using `text-embedding-004`  
- FAISS vector store for fast similarity search  
- Gemini 2.5 Flash for natural and high-quality answers  
- Clean and modern Gradio interface  


## 🧠 How It Works

1. You enter URLs (one per line)  
2. The system fetches and cleans the HTML  
3. Text is split into small chunks  
4. Each chunk is embedded  
5. Stored inside a FAISS vector database  
6. When you ask a question, the most relevant chunks are retrieved  
7. Gemini answers based only on the retrieved context  


## 📦 Installation

### 1. Install dependencies
pip install -r requirements.txt

### 2. Add your Google Gemini API key
**Linux/macOS**
export GEMINI_API_KEY="YOUR_KEY"

**Windows PowerShell**powershell
setx GEMINI_API_KEY "YOUR_KEY"



## 📂 Project Structure

.
├── app.py
├── app.ipynb
├── requirements.txt
├── README.md
└── images/
    ├── screenshot1.png
    └── screenshot2.png


## 📝 Notes

- Works best on clean, text-rich websites  
- Scanned/graphical pages will not extract properly  
- You can extend this project to support PDFs, YouTube transcripts, etc.  


## ⭐ Enjoy

If you like the project, feel free to star the repo or build on it!
