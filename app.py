import os
from dotenv import load_dotenv
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
load_dotenv()
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)
db = Chroma(
    collection_name="document_qa_collection",
    embedding_function=embedding_function,
    persist_directory="chroma_persistent_storage"
)

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as f:
                documents.append(f.read())
    return documents

def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks
directory_path = "./news_articles"
documents = load_documents_from_directory(directory_path)
all_chunks = []
for doc in documents:
    all_chunks.extend(split_text(doc))
db.add_texts(all_chunks)

def query_documents(question, n_results=2):
    results = db.similarity_search(question, k=n_results)
    return [res.page_content for res in results]

def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt_template = PromptTemplate.from_template(
        "You are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the answer concise."
        "\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )
    formatted_prompt = prompt_template.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=formatted_prompt)])
    return response.content

question = "tell me about surya"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)

print(answer)
