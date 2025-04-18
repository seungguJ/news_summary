from crawling import news_crawling
from crawling_yahoo import yahoo_finance_news_crawling
from EEVE_RAG import EEVE_RAG
import datetime
from zoneinfo import ZoneInfo
import os

def load_query_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

today = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y-%m-%d')
filename = f"./News/{today}_news.txt"  # Save as text file
yahoo_fiance_filename = f"./News/{today}_yahoo_finance_news.txt"
if os.path.exists(filename):
    pass
else:
    news_crawling()

if os.path.exists(yahoo_fiance_filename):
    pass
else:
    yahoo_finance_news_crawling()

EEVE_RAG(yahoo_fiance_filename, translate=True, llm_model_name="exaone")
EEVE_RAG(filename, llm_model_name="exaone")