import requests
from bs4 import BeautifulSoup
import datetime
from zoneinfo import ZoneInfo
import os

headers = {
    "User-Agent": "Your User Agent"
}

def create_soup(url):
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")
    return soup

def write_news_to_text_file(filename, news_data):
    with open(filename, 'w', encoding='utf-8') as file:
        for news_title, news_content in news_data:
            file.write(f"- 제목: {news_title}\n  본문: {news_content}\n")
        file.write('\n')

def get_article(content):
    try:
        article_content = ""
        soup = create_soup(content)  # 해당 URL로부터 BeautifulSoup 객체 생성
        content = soup.find("div", attrs={"class": "body yf-tsvcyu"}).find_all("p", attrs={"class": "yf-1pe5jgt"})  # 본문 영역
        for element in content:
            # p 태그의 상위 li 태그 중 class가 "yf-1woyvo2"가 있는지 확인
            if not element.find_parent("li", class_="yf-1woyvo2"):
                article_content += element.get_text(strip=True)

        if article_content != "":
            return article_content  # 본문 텍스트 추출
        return "본문을 가져올 수 없습니다."
    except Exception as e:
        return f"에러 발생: {e}"


def yahoo_finance_news_crawling():
    today = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y-%m-%d')
    file_dir = "./News"
    filename = f"{file_dir}/{today}_yahoo_finance_news.txt"  # Save as text file
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Directory '{file_dir}' created.")
    else:
        pass

    url = "https://finance.yahoo.com/news/"
    soup = create_soup(url)

    # h2 news
    h2_news = soup.find_all("div", attrs={"class": "content yf-1jvnfga btmMarginMd"})

    news_list = []

    for i in range(len(h2_news)):
        news_title = h2_news[i].get_text()
        news_link = h2_news[i].a["href"]
        news_content = get_article(news_link)
        news_list.append((news_title, news_content))
    
    
    # h3 news
    h3_news = soup.find_all("div", attrs={"class": "content yf-82qtw3"})

    for i in range(len(h3_news)):
        news_title = h3_news[i].get_text()
        news_link = h3_news[i].a["href"]
        news_content = get_article(news_link)
        if "에러 발생" in news_content:
            continue
        news_list.append((news_title, news_content))
    
    write_news_to_text_file(filename, news_list)
    print("News written to", filename)


if __name__ == "__main__":
    yahoo_finance_news_crawling()
