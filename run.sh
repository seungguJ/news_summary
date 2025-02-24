#!/bin/bash

# TODAY=$(TZ="Asia/Seoul" date +"%Y-%m-%d")
# echo "Today's date (Asia/Seoul): $TODAY"

# # ollama serve를 백그라운드에서 실행
# ollama serve &
# # 서비스가 시작될 시간을 고려하여 잠시 대기
# sleep 5

# # 메인 뉴스 요약 스크립트 실행
# python main_news_EEVE.py

# git에 변경사항 추가 (오늘 날짜에 해당하는 파일)
git add "./News_Summaries_EEVE/${TODAY}.txt"

# 커밋 메시지 작성 후 커밋
git commit -m "News summary update"

# 원격 저장소에 push
git push origin master
