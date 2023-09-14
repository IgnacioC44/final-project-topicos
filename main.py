import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from PyPDF2 import PdfReader

load_dotenv()

#using openai api key and loading data
openai.api_key = os.getenv("OPENAI_API_KEY")

#for pdfs
# loader_pdf = PyPDFLoader("./data/Recipes.pdf")
# pages = loader_pdf.load()

# len(pages)

# page = pages[4]
data_path = "./data/"

pdf_files= os.listdir(data_path)
print(pdf_files)

def get_pdf_text(data_path, pdf_files):
    
    text = ""

    for pdf_file in pdf_files:
        reader = PdfReader(data_path+pdf_file)
        for page in reader.pages:
            text += page.extract_text()

    return text

text = get_pdf_text(data_path, pdf_files)


def get_chunk_text(text):
    
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 500,
    chunk_overlap = 100,
    length_function = len
    )

    chunks = text_splitter.split_text(text)

    return chunks

# print(get_chunk_text(text))
data = get_chunk_text(text)

#print(page)

#for csv's

#loader_csv = CSVLoader(file_path="./data/RecipeNLG_dataset.csv", encoding="utf8")

# csvLoader = loader_csv[1].load()
# csvLoader1 = csvLoader[:5000]
# print(len(csvLoader1))

# data = loader_pdf.load()

embeddings = OpenAIEmbeddings()

index_creator = VectorstoreIndexCreator()

docsearch = index_creator.from_loaders([data])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

query = "Recommend me a soup recipe which contains tomato and give me the ingredients and the instructions"
response = chain({"question": query})

print(response['result'])