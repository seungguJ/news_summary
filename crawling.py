import requests
from bs4 import BeautifulSoup
import datetime
import os

headers = {
    "User-Agent": "Your User Agent"  # User Agent 정보 입력
}

def create_soup(url):
    res = requests.get(url, headers=headers)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "lxml")
    return soup

def write_news_to_text_file(filename, news_data):
    with open(filename, 'w', encoding='utf-8') as file:
        for section, news_list in news_data.items():
            for news_title, news_content in news_list:
                file.write(f"- 제목: {news_title}\n  본문: {news_content}\n")
            file.write('\n')

def get_article_content(url):
    try:
        soup = create_soup(url)  # 해당 URL로부터 BeautifulSoup 객체 생성
        content = soup.find("div", attrs={"class": "newsct_article _article_body"})  # 네이버 뉴스 본문 영역
        if content:
            return content.get_text(strip=True)  # 본문 텍스트 추출
        return "본문을 가져올 수 없습니다."
    except Exception as e:
        return f"에러 발생: {e}"
    
def get_article_content_from_aitimes(url):
    try:
        soup = create_soup(url)  # 해당 URL로부터 BeautifulSoup 객체 생성
        content = soup.find("div", attrs={"class": "article-body"})  # AI 뉴스 본문 영역
        if content:
            return content.get_text(separator=" ", strip=True)  # 본문 텍스트 추출 및 줄바꿈 제거
        return "본문을 가져올 수 없습니다."
    except Exception as e:
        return f"에러 발생: {e}"


def news_crawling():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    file_dir = "./News"
    filename = f"{file_dir}/{today}_news.txt"  # Save as text file
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        print(f"Directory '{file_dir}' created.")
    else:
        pass
    news_data = {}

    # 언론사 6개당 헤드라인 뉴스 3개
    url = "https://news.naver.com/main/ranking/popularDay.naver"
    soup = create_soup(url)
    whole_news = soup.find("div", attrs={"class": "_officeCard _officeCard0"})

    for i in range(6):
        section_name = whole_news.find_all("div", attrs={"class": "rankingnews_box"})[i].find("strong",
                                                                                                attrs={
                                                                                                    "class": "rankingnews_name"}).get_text()
        section_news_list = []
        for news_item in whole_news.find_all("div", attrs={"class": "rankingnews_box"})[i].find_all(
                "a", attrs={"class": "list_title nclicks('RBP.rnknws')"}):
            news_title = news_item.get_text()  # 뉴스 제목 추출
            news_link = news_item["href"]  # 뉴스 링크 추출
            news_content = get_article_content(news_link)  # 본문 내용 크롤링
            # section_news_list.append((news_title, news_content, news_link))
            section_news_list.append((news_title, news_content)) # 링크 제거
        news_data[section_name] = section_news_list

    # 경제 뉴스
    url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101"
    soup = create_soup(url)
    eco_news = soup.find_all("ul", attrs={"class": "sa_list"})
    eco_news_list = []
    for i in range(6):
        for news_item in eco_news[i].find_all("li", attrs={"class": "sa_item"}):
            eco_new = news_item.find("a", attrs={"class": "sa_text_title"}).get_text()
            eco_new_link = news_item.a["href"]
            eco_new_content = get_article_content(eco_new_link)  # 본문 내용 크롤링
            # eco_news_list.append((eco_new, eco_new_content, eco_new_link))
            eco_news_list.append((eco_new, eco_new_content)) # 링크 제거
    news_data["경제 뉴스"] = eco_news_list

    # IT/과학 뉴스
    url = "https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=105"
    soup = create_soup(url)
    it_news = soup.find_all("a", attrs={"class": "sa_text_title"})
    it_news_list = []
    for news_item in it_news:
        it_title = news_item.get_text()
        it_link = news_item["href"]
        it_content = get_article_content(it_link)  # 본문 내용 크롤링
        # it_news_list.append((it_title, it_content, it_link))
        it_news_list.append((it_title, it_content)) # 링크 제거
    news_data["IT/과학 뉴스"] = it_news_list

    # AI 뉴스
    url = "https://www.aitimes.com/"
    soup = create_soup(url)
    ai_news = soup.find_all("div", attrs={"class": "item"})
    ai_news_list = []
    num_news = 0
    for news_item in ai_news:
        ai_title = news_item.get_text()
        ai_link = news_item.a["href"]
        ai_link = url + ai_link[1:]
        ai_content = get_article_content_from_aitimes(ai_link)  # 본문 내용 크롤링
        # ai_news_list.append((ai_title, ai_content, ai_link))
        ai_news_list.append((ai_title, ai_content)) # 링크 제거
        num_news += 1
        if num_news == 20:
            break
    news_data["AI 뉴스"] = ai_news_list

    write_news_to_text_file(filename, news_data)
    print("News written to", filename)
    return filename
