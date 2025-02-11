# News Summary

## How to run
```
python main_news_EEVE.py
```

### Information

<details>
<summary> EEVE_RAG.py </summary>

similarity_threshold: A hyperparameter for the similarity score. 

A lower value indicates higher similarity

template: prompt template

</details>

<details>
<summary> crawling.py </summary>

headers = {"User-Agent": "Your user-agent"}

You can find your user-agent from <https://www.whatismybrowser.com/detect/what-is-my-user-agent>

</details>

### Crawling

- News headlines from 6 media companies on Naver
- Economics articles from naver
- IT/Science articles from naver
- AI articles from AI times

### Citation
<https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0>

```
@misc{kim2024efficient,
      title={Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models}, 
      author={Seungduk Kim and Seungtaek Choi and Myeongho Jeong},
      year={2024},
      eprint={2402.14714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```