import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv

#using openai api key and loading data
os.environ["OPEN_API_KEY"] = "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"
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPEN_API_KEY']

