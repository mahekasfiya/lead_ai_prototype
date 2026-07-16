from dotenv import load_dotenv
import os

load_dotenv()

SERPAPI_KEY= os.getenv("SERPAPI_KEY")
OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")