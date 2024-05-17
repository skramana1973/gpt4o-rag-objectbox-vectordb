# Import necessary libraries and modules
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
import pprint

# Load environment variables from .env file
load_dotenv()
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load data from the specified URL
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
data = loader.load()
print("Data loaded from the URL:")
print(data)

# Import ObjectBox for vector database and OpenAIEmbeddings for embeddings
from langchain_objectbox.vectorstores import ObjectBox
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Split the loaded documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
print("Documents after splitting:")
print(documents)

# Create a vector store from the documents using ObjectBox and OpenAIEmbeddings
vector = ObjectBox.from_documents(
    documents, OpenAIEmbeddings(), embedding_dimensions=768
)
print("Vector store created:")
print(vector)

# Initialize the GPT-4o model
llm = ChatOpenAI(model="gpt-4o")
print("GPT-4o model initialized.")

# Pull the prompt template from the hub
prompt = hub.pull("rlm/rag-prompt")
print("Prompt template pulled from the hub:")
print(prompt)

# Create a RetrievalQA chain with the specified LLM, retriever, and prompt
qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=vector.as_retriever(), chain_type_kwargs={"prompt": prompt}
)
print("RetrievalQA chain created.")


# Define a function to ask a question and print the result
def ask_question(question):
    result = qa_chain({"query": question})
    print(f"Question: {question}")
    print("Result:")
    pp = pprint.PrettyPrinter(indent=5)
    pp.pprint(result["result"])


# Ask questions and print the results
ask_question("Explain what is langsmith")
ask_question("Explain Monitoring and A/B Testing in langsmith")
ask_question("Explain How to enable langsmith and start using it")
