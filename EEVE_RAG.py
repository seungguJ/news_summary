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

def count_tokens(text, model="gpt-3.5-turbo"):
    """텍스트의 토큰 수를 계산합니다."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def print_token_usage(query, documents, model="gpt-3.5-turbo"):
    """쿼리와 문서의 토큰 수를 출력하고 총합을 계산합니다."""
    query_tokens = count_tokens(query, model)
    print(f"Query tokens: {query_tokens}")

    total_doc_tokens = 0
    for i, doc in enumerate(documents):
        doc_tokens = count_tokens(doc.page_content, model)
        total_doc_tokens += doc_tokens
        print(f"Document {i+1}: {doc_tokens} tokens")

    total_tokens = query_tokens + total_doc_tokens
    print(f"Total input tokens: {total_tokens}")

    # 최대 토큰 수 설정 (예: 4096 또는 8192)
    max_tokens = 4096  # 필요에 따라 변경
    print(f"Max tokens allowed: {max_tokens}")

    if total_tokens > max_tokens:
        print("⚠️ Warning: Total tokens exceed the model's maximum input limit!")
    else:
        print("✅ Total tokens are within the acceptable range.")

def split_articles_by_title(text):
    # '- 제목'을 기준으로 텍스트 분할
    articles = text.split('- 제목')

    # 첫 번째 요소는 빈 문자열일 수 있으므로 제거
    articles = [article.strip() for article in articles if article.strip()]

    # 분할된 기사에 '- 제목' 다시 추가
    articles_with_title = ['- 제목' + article for article in articles]
    
    return articles_with_title

def preprocess_result(text):
    lines = text.strip().splitlines()
    processed_lines = []

    for line in lines:
        if "제목" in line:
            clean_line = line.replace("#","").strip()
            processed_lines.append(f"\n ### {clean_line}  \n\n ")
        elif "요약:" in line:
            clean_line = line.replace("**","").strip()
            clean_line = clean_line.replace("#","").strip()
            processed_lines.append((clean_line.replace("요약", "**요약**")))
        else:
            processed_lines.append(line)
    
    result_text = "".join(processed_lines)
    return result_text

def EEVE_RAG(filename, translate=False):
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    llm = ChatOllama(
        model="EEVE-Korean-10.8B:latest",
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

        similarity_threshold = 3. ### (HYPER PARAMETER) Threshold 값 변경
        is_duplicate = False

        if search_results:
            similarity_score = search_results[0][1]
            if similarity_score <= similarity_threshold:
                print(f"Article {i+1} is a duplicate. Skipping...")
                is_duplicate = True
        
        if not is_duplicate:
            # 토큰 수 확인
            tokens = count_tokens(doc.page_content, model="gpt-3.5-turbo")
            print(f"Article {i+1} Tokens: {tokens}")

            # LLM을 통한 요약 생성
            result = llm_chain.run(content=doc.page_content)
            result = preprocess_result(result)
            if translate:
                all_results += f"## **Financial Article {number+1} Summary**:\n{result}\n\n"
            else:
                all_results += f"## **Article {number+1} Summary**:\n{result}\n\n"

            # 벡터 DB에 요약 결과 저장 (중복 방지를 위해)
            vector_db.add_texts([result], metadatas=[{"source": f"Article_{i+1}"}])
            vector_db.persist()
            number += 1

    # 결과 저장
    output_dir = "./News_Summaries_EEVE"
    output_filename = f"{output_dir}/{today}.md"

    # 디렉터리가 존재하지 않으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    else:
        pass
    with open(output_filename, 'a', encoding='utf-8') as file:
        file.write(all_results)
    
    print(f"All summaries saved to {output_filename}")
    print(f"Vector DB saved to {vector_db_dir}")