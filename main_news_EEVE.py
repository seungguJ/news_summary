from crawling import news_crawling
from EEVE_RAG import EEVE_RAG
import datetime
import os

def load_query_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

today = datetime.datetime.today().strftime('%Y-%m-%d')
filename = f"./News/{today}_news.txt"  # Save as text file
if filename in os.listdir():
    pass
else:
    filename = news_crawling()

EEVE_RAG(filename)