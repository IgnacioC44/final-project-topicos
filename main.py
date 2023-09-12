import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

#using openai api key and loading data
openai.api_key = os.getenv("OPENAI_API_KEY")

#for pdfs
loader_pdf = PyPDFLoader("./data/recipe-book-1.zp210082.pdf")
pages = loader_pdf.load()

len(pages)

page = pages[4]
#print(page)

#for csv's

loader_csv = CSVLoader(file_path="./data/RecipeNLG_dataset.csv", encoding="utf8")

data = loader_csv.load()

embeddings = OpenAIEmbeddings()

index_creator = VectorstoreIndexCreator()

docsearch = index_creator.from_loaders([loader_csv])

chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

query = "Give me the Cheeseburger Potato Soup ingredients"
response = chain({"question": query})

print(response['result'])