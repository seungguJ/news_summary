# News Summary

## How to run
```
bash run.sh
```

### Information

<details>
<summary> EEVE_RAG.py </summary>

similarity_threshold: A hyperparameter for the similarity score. 

A lower value indicates higher similarity

template: prompt template

Add EXAONE-3.5-7.8B-Instruct-BF16 model
</details>

<details>
<summary> crawling file </summary>

headers = {"User-Agent": "Your user-agent"}

You can find your user-agent from <https://www.whatismybrowser.com/detect/what-is-my-user-agent>

</details>

### Crawling

- Financial articles from yahoo finance
- News headlines from 6 media companies on Naver
- Economics articles from naver
- IT/Science articles from naver
- AI articles from AI times

### Citation
<https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0>
<https://github.com/LG-AI-EXAONE/EXAONE-3.5?tab=readme-ov-file>

```
@misc{kim2024efficient,
      title={Efficient and Effective Vocabulary Expansion Towards Multilingual Large Language Models}, 
      author={Seungduk Kim and Seungtaek Choi and Myeongho Jeong},
      year={2024},
      eprint={2402.14714},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@article{exaone-3.5,
  title={EXAONE 3.5: Series of Large Language Models for Real-world Use Cases},
  author={LG AI Research},
  journal={arXiv preprint arXiv:2412.04862},
  year={2024}
}
```