import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv()

#using openai api key and loading data
openai.api_key = os.getenv("OPEN_API_KEY")

