from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import gradio as gr

# This block of code is to clean the URLs
from bs4 import BeautifulSoup
import requests


def clean_url_text(content):

    soup = BeautifulSoup(content, "html.parser")

    # Remove scripts and styles
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = " ".join(text.split())  # remove extra spaces/newlines

    return text

# defining the llm
API_KEY = "SECRET"
llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=API_KEY,
        temperature=0.2,
      )

embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=API_KEY,
            )

retrievalQA = None
retriever = None

PROMPT_TEMPLATE = """
Use the following context to answer the question.

Rules:
1. If the answer is not in the context, say: "I can't find the final answer."
2. Give the final answer in 1–2 natural sentences.
3. Do NOT show your reasoning or steps.
4. Do NOT be overly specific. Prefer natural phrasing like "the United States" instead of "U.S."
5. You may add one short extra detail from the context to make the answer fuller.
6. Do NOT over-explain or summarize the entire context.

Context:
{context}

Question: {question}

Final Answer (1–2 natural sentences):
"""


PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])

# header for wikipedea
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

# function for loading URLS
def load_url(urls: list):
    try:
        loader = UnstructuredURLLoader(urls=urls, headers=headers)
        docs = loader.load()

        # ✅ Apply cleaning here (you wrote the function, now we use it)
        for d in docs:
            d.page_content = clean_url_text(d.page_content)

    except Exception as e:
        return f"❌ Error loading URLs: {e}"

    return docs

# function for chunking data
def chunk_data(docs):
    # if docs is an error string, stop early
    if isinstance(docs, str):
        return docs

    try:
        chuncker = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=30
        )

        # ✅ split ONCE, not inside a loop
        chunks = chuncker.split_documents(docs)

    except Exception as e:
        return f"❌ Error chunking URLs: {e}"

    return chunks

# function for embedding content and storing them in vector store
def create_vectore_store(chunks):
    global retriever

    # if chunks is an error string, stop early
    if isinstance(chunks, str):
        return chunks

    try:
        vectore_store = FAISS.from_documents(chunks, embeddings)

        # ✅ store retriever globally so answer_question can use it
        retriever = vectore_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    except Exception as e:
        return f"❌ Error creating vector_store URLs: {e}"

    return vectore_store

# function for answering question
def answer_question(question):
    global retrievalQA, retriever

    if retriever is None:
        return "⚠️ Please load URLs and create the vector store first."

    # ✅ RetrievalQA is a class, so we call it normally
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = chain({"query": question})

    # ✅ return the final answer text
    return result["result"]

# creating the final functions for gradio
def process_urls(url_text):
    """
    url_text: multiline textbox (one URL per line)
    """
    global retrievalQA, retriever

    urls = [u.strip() for u in url_text.split("\n") if u.strip()]
    if not urls:
        return "⚠️ Please paste at least one URL."

    docs = load_url(urls)
    if isinstance(docs, str):
        return docs

    chunks = chunk_data(docs)
    if isinstance(chunks, str):
        return chunks

    store = create_vectore_store(chunks)
    if isinstance(store, str):
        return store

    return "✅ URLs processed successfully! You can now ask questions."


def ask_ui(q):
    return answer_question(q)

# creating the gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🌐 RAG Link Q&A Assistant (Gemini 2.5 Flash)")
    gr.Markdown("Paste one or more links below (one per line), process them, then ask questions.")

    with gr.Row():
        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 1) Paste URLs & Process")
            url_input = gr.Textbox(
                label="URLs (one per line)",
                placeholder="https://en.wikipedia.org/wiki/Elon_Musk\nhttps://example.com/page",
                lines=8
            )
            process_btn = gr.Button("Process URLs", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2, min_width=520):
            gr.Markdown("### 2) Ask a Question")
            question = gr.Textbox(label="Your Question", placeholder="Type here...")
            answer = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=12,
                max_lines=35,
                show_copy_button=True
            )

    process_btn.click(process_urls, inputs=url_input, outputs=status)
    question.submit(ask_ui, inputs=question, outputs=answer)

demo.queue()
demo.launch(debug=True)