from langchain_ollama import ChatOllama
from langchain_core.callbacks.manager import CallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import datetime
from zoneinfo import ZoneInfo
import tiktoken
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch
import sentencepiece as spm

def count_tokens(text, model="gpt-3.5-turbo"): # need to be changed
    """텍스트의 토큰 수를 계산합니다."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def split_text_by_tokens(text, max_tokens=2048, model="gpt-3.5-turbo"):
    """텍스트를 토큰 제한에 맞게 분할합니다."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return [text]
    else:
        return [text[:max_tokens]]
    
    # 문단 단위로 분할
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for paragraph in paragraphs:
        paragraph_tokens = len(encoding.encode(paragraph))
        
        if current_tokens + paragraph_tokens > max_tokens:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
            
            # 단일 문단이 max_tokens를 초과하는 경우
            if paragraph_tokens > max_tokens:
                # 문장 단위로 분할
                sentences = paragraph.split('. ')
                current_sentences = []
                current_sentence_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = len(encoding.encode(sentence))
                    if current_sentence_tokens + sentence_tokens > max_tokens:
                        if current_sentences:
                            chunks.append('. '.join(current_sentences) + '.')
                            current_sentences = []
                            current_sentence_tokens = 0
                    current_sentences.append(sentence)
                    current_sentence_tokens += sentence_tokens
                
                if current_sentences:
                    chunks.append('. '.join(current_sentences) + '.')
            else:
                current_chunk.append(paragraph)
                current_tokens = paragraph_tokens
        else:
            current_chunk.append(paragraph)
            current_tokens += paragraph_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def split_articles_by_title(text):
    # '- 제목'을 기준으로 텍스트 분할
    articles = text.split('- 제목')

    # 첫 번째 요소는 빈 문자열일 수 있으므로 제거
    articles = [article.strip() for article in articles if article.strip()]

    # 분할된 기사에 '- 제목' 다시 추가
    articles_with_title = ['- 제목' + article for article in articles]
    
    return articles_with_title

def is_format_valid(text):
    if "제목" in text and "요약" in text:
        return True
    else:
        return False

def preprocess_result(text):
    # Remove markdown code block markers
    text = text.replace("```markdown", "").replace("```", "").strip()
    
    # Remove the last ``` if it exists
    if text.endswith("```"):
        text = text[:-3].strip()
    
    lines = text.split('\n')
    processed_lines = []
    current_title = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 제목 처리
        if "### 제목:" in line:
            # 제목과 요약이 한 줄에 있는 경우
            if "요약" in line:
                parts = line.split("요약")
                title_part = parts[0].replace("### 제목:", "").strip()
                summary_part = parts[1].strip()
                
                # 제목 처리
                title_line = f"### 제목: {title_part}"
                title_line = title_line.replace("**", "")
                title_line = title_line.replace("~", r"\~")
                processed_lines.append(title_line)
                processed_lines.append("")  # 1줄 개행
                
                # 요약 처리
                if summary_part:
                    summary_part = summary_part.replace("*", "").replace("#","")
                    summary_part = f'**요약**: {summary_part}'
                    summary_part = summary_part.replace(": :", ":")
                    summary_line = summary_part.replace("~", r"\~")
                    processed_lines.append(summary_line)
            else:
                # 현재 줄에 제목이 있는지 확인
                current_title = line.replace("### 제목:", "").strip()
                if current_title:  # 현재 줄에 제목이 있는 경우
                    title_line = f"### 제목: {current_title}"
                    title_line = title_line.replace("~", r"\~")
                    processed_lines.append(title_line)
                    processed_lines.append("")  # 1줄 개행
                else:  # 제목이 다음 줄에 있는 경우
                    if i + 1 < len(lines) and not lines[i + 1].strip().startswith("###") and not lines[i + 1].strip().startswith("요약"):
                        next_line = lines[i + 1].strip()
                        title_line = f"### 제목: {next_line}"
                        title_line = title_line.replace("~", r"\~")
                        processed_lines.append(title_line)
                        processed_lines.append("")  # 1줄 개행
                        i += 1  # 다음 줄을 건너뛰기

        # 요약 처리
        elif "요약:" in line:
            clean_line = line.replace("**", "").replace("#", "").strip()
            summary_line = clean_line.replace("요약", " **요약**")
            summary_line = summary_line.replace("~", r"\~")
            processed_lines.append(summary_line)

        # 일반 텍스트 처리
        else:
            clean_line = line.replace("~", r"\~")
            processed_lines.append(clean_line)
        
        i += 1

    return '\n'.join(processed_lines)

def process_chunks_with_llm(chunks, llm_chain):
    """여러 청크를 하나의 요약으로 처리합니다."""
    combined_summary = ""
    
    # 각 청크에 대한 간단한 요약 생성
    chunk_summaries = []
    for chunk in chunks:
        result = llm_chain.run(content=chunk)
        flag = is_format_valid(result)
        if not flag:
            while not flag:
                print(f"Invalid format. Retrying...")
                result = llm_chain.run(content=chunk)
                flag = is_format_valid(result)
        # for debug
        if flag:
            with open("unprocessed_result.txt", 'a', encoding='utf-8') as file:
                file.write(result)
        return preprocess_result(result)
    
    # # 모든 청크 요약을 하나로 합침
    # combined_content = "\n\n".join(chunk_summaries)
    
    # # 최종 요약 생성
    # final_result = llm_chain.run(content=combined_content)
    # flag = is_format_valid(final_result)
    # if not flag:
    #     while not flag:
    #         final_result = llm_chain.run(content=combined_content)
    #         flag = is_format_valid(final_result)
    
    # return preprocess_result(final_result)

def EEVE_RAG(filename, output_dir, translate=False, llm_model_name='EEVE', is_test=False):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    if llm_model_name == 'EEVE':
        llm = ChatOllama(
            model="EEVE-Korean-10.8B:latest",
            callback_manager=CallbackManager([]),
        )
    elif llm_model_name == "exaone":
        llm = ChatOllama(
            model="exaone3.5-instruct-7.8B:latest",
            callback_manager=CallbackManager([]),
        )

    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cuda:0'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    filepath = os.path.join(os.getcwd(), filename)
    documents = TextLoader(filepath).load()

    # 전체 텍스트를 하나로 결합 후 기사 분할
    full_text = "\n".join([doc.page_content for doc in documents])
    all_articles = split_articles_by_title(full_text)
    
    # 분할된 기사들을 Document 객체로 변환
    split_documents = [Document(page_content=article) for article in all_articles]

    today = datetime.datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y-%m-%d')
    vector_dir = "./vector_db_EEVE"
    vector_db_dir = f"{vector_dir}/{today}"
    if not os.path.exists(vector_dir):
        os.makedirs(vector_dir)
        print(f"Directory '{vector_dir}' created.")
    else:
        pass

    if os.path.exists(vector_db_dir):
        vector_db = Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)
    else:
        vector_db = Chroma(persist_directory=vector_db_dir, embedding_function=embedding_model)
        vector_db.persist()  # 빈 DB 초기화

    # 프롬프트 템플릿
    if not translate:
        template = (
            "다음 기사를 읽고 다음과 같은 한 개의 '제목'과 '요약'으로 구성된 마크다운(Markdown) 형식으로 요약해 주세요:\n"
            "{content}\n"
            "요약 형식(마크다운):\n"
            "### 제목: [제목]\n"
            "**요약**: [요약문]\n\n"
            "요약 작성 시 유의사항:\n"
            "1. 각 요약문 앞서 주어진 마크다운 요약 형식에 맞춰서 요약해주세요. 제목 다음 줄바꿈 한 뒤 요약으로 넘어가주세요.\n"
            "2. 각 요약문은 해당 기사 본문의 핵심 내용을 모두 포함해야 합니다.\n"
            "3. 요약문은 구체적이고 자세하게 작성하며, 최대 5 줄의 문장으로 작성해 주시되, 충분한 정보를 제공해 주세요.\n"
            "4. 기사에 언급된 인물, 사건 발생 시간, 장소, 사건의 경위 및 경찰의 조사 상황 등 중요한 세부 사항을 빠뜨리지 말고 포함해 주세요.\n"
            "5. 각 요약문은 명확하고 간결한 문장으로 한 개의 제목과 요약으로 구성해 주세요.\n"
            "6. 만약 기사 내용이 잘못되었거나 누락되었으면 요약문에 포함하지 않아주세요."
        )
    else:
        template = (
            "다음 기사를 읽고 다음과 같은 한 개의 '제목'과 '요약'으로 구성된 마크다운(Markdown) 형식으로 한국말로 요약해 주세요:\n"
            "{content}\n"
            "요약 형식(마크다운):\n"
            "### 제목: [제목]\n"
            "**요약**: [요약문]\n\n"
            "요약 작성 시 유의사항:\n"
            "1. 각 요약문 앞서 주어진 요약 형식에 맞춰서 요약해주세요.제목 다음 줄바꿈 한 뒤 요약으로 넘어가주세요.\n"
            "2. 각 요약문은 해당 기사 본문의 핵심 내용을 모두 포함해야 합니다.\n"
            "3. 요약문은 구체적이고 자세하게 작성하며, 최대 5 줄의 문장으로 작성해 주시되, 충분한 정보를 제공해 주세요.\n"
            "4. 기사에 언급된 인물, 사건 발생 시간, 장소, 사건의 경위 및 경찰의 조사 상황 등 중요한 세부 사항을 빠뜨리지 말고 포함해 주세요.\n"
            "5. 각 요약문은 명확하고 간결한 문장으로 한 개의 제목과 요약으로 구성해 주세요.\n"
            "6. 만약 기사 내용이 잘못되었거나 누락되었으면 요약문에 포함하지 않아주세요."
        )

    prompt = PromptTemplate(template=template, input_variables=["content"])

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # 각 기사별로 요약 생성 및 결과 저장
    all_results = ""
    number = 0
    for i, doc in enumerate(split_documents):
        print(f"Processing Article {i+1}...")

        query_embedding = embedding_model.embed_query(doc.page_content)
        search_results = vector_db.similarity_search_by_vector_with_relevance_scores(query_embedding, k=1)

        similarity_threshold = 3.5 ### (HYPER PARAMETER) Threshold 값 변경
        is_duplicate = False

        if search_results:
            similarity_score = search_results[0][1]
            if similarity_score <= similarity_threshold:
                print(f"Article {i+1} is a duplicate. Skipping...")
                is_duplicate = True
        
        if not is_duplicate:
            article_summary = llm_chain.run(content=doc.page_content)
            article_summary = preprocess_result(article_summary)
            # 토큰 수 확인 및 분할
            # chunks = split_text_by_tokens(doc.page_content)
            
            # 모든 청크를 하나의 요약으로 처리
            # article_summary = process_chunks_with_llm(chunks, llm_chain)
            
            
            if translate:
                all_results += f"## **Financial Article {number+1} Summary**:\n{article_summary}\n\n"
            else:
                all_results += f"## **Article {number+1} Summary**:\n{article_summary}\n\n"

            # 벡터 DB에 요약 결과 저장 (중복 방지를 위해)
            if not is_test:
                vector_db.add_texts([article_summary], metadatas=[{"source": f"Article_{i+1}"}])
                vector_db.persist()
            number += 1

    # 디렉터리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        pass
    if not is_test:
        output_filename = f"{output_dir}/{today}.md"
        with open(output_filename, 'a', encoding='utf-8') as file:
            file.write(all_results)
    
        print(f"All summaries saved to {output_filename}")
        print(f"Vector DB saved to {vector_db_dir}")
    # else:
    #     with open("test_output.txt", 'w', encoding='utf-8') as file:
    #         file.write(all_results)
    #     print(f"All summaries saved to test_output.txt")