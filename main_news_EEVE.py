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
month = datetime.datetime.now().strftime('%Y-%m')
filename = f"./News/{month}/{today}_news.txt"  # Save as text file
yahoo_fiance_filename = f"./News/{month}/{today}_yahoo_finance_news.txt"
if os.path.exists(filename):
    pass
else:
    news_crawling(filename)

if os.path.exists(yahoo_fiance_filename):
    pass
else:
    yahoo_finance_news_crawling(yahoo_fiance_filename)

news_dir = f"./News_Summaries_EEVE/{month}"
news_filename = f"./News_Summaries_EEVE/{month}/{today}.md"
if os.path.exists(news_filename):
    print("Already Exists")
    pass
else:
    EEVE_RAG(yahoo_fiance_filename, news_dir, translate=True, llm_model_name="exaone")
    EEVE_RAG(filename, news_dir, llm_model_name="exaone")