import numpy as np
from openai import OpenAI
import pdfplumber
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

client = OpenAI()

INDEX_NAME = "embeddings-index"           # Redisearch에서 사용할 index
PREFIX = "doc"                            
DISTANCE_METRIC = "COSINE"                # 텍스트 데이터의 embedding이므로

# Redis-Stack-Server 관련 설정
REDIS_HOST = "localhost"
REDIS_PORT = 6380
REDIS_PASSWORD = "" 

class DataService():
    # Redis-Stack-Server연결(RedisSearch기능을 사용하기 위함)
    def __init__(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD
        )

    # embeddings-index를 가진 Redis 데이터 삭제
    def drop_redis_data(self, index_name: str = INDEX_NAME):
        try:
            self.redis_client.ft(index_name).dropindex()
            #print('Index dropped')
        except:
            print('Index does not exist')

    # 주어진 임베딩을 Redis에 저장
    def load_data_to_redis(self, embeddings):
        vector_dim = len(embeddings[0]['vector'])  # 벡터 차원
        vector_number = len(embeddings) # 벡터의 개수
        text = TextField(name="text") # 텍스트 필드를 정의
        text_embedding = VectorField("vector", # 벡터 필드를 정의
                                     "FLAT", {
                                         "TYPE": "FLOAT32",
                                         "DIM": vector_dim,
                                         "DISTANCE_METRIC": "COSINE",
                                         "INITIAL_CAP": vector_number,
                                     })
        fields = [text, text_embedding]
        # 사실 이 try-catch는 drop_redis_data를 수행했다면 의미가 없음
        try:
            self.redis_client.ft(INDEX_NAME).info() # 인덱스의 정보를 조회하여 인덱스가 존재하는지 확인
            print("Index already exists")
        except:
            # 인덱스가 존재하지 않을 경우, 새 인덱스를 생성
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=fields,
                definition=IndexDefinition(
                    prefix=[PREFIX], index_type=IndexType.HASH)
            )
        for i, embedding in enumerate(embeddings):
            key = f"{PREFIX}:{str(embedding['id'])}" 
            # print(f"Storing document {i + 1} with key {key} and text: {embedding['text'][:100]}...")
            embedding["vector"] = np.array(  # 벡터를 np.float32 타입으로 변환한 후, 바이트 문자열로 변환
                embedding["vector"], dtype=np.float32).tobytes()
            self.redis_client.hset(key, mapping=embedding)  # 해시 데이터 구조(key-embedding)로 저장
        print(
            f"Loaded {self.redis_client.info()['db0']['keys']} documents in Redis search index with name: {INDEX_NAME}")

    # PDF에서 텍스트 추출 및 임베딩 생성(텍스트 데이터를 벡터 공간에 매핑)
    def pdf_to_embeddings(self, pdf_path: str, chunk_length: int = 1000):
        chunks = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_page = page.extract_text() 
                # print(f"Extracted text from page: {text_page[:100]}...") 
                chunks.extend([text_page[i:i + chunk_length].replace('\n', '')
                               for i in range(0, len(text_page), chunk_length)])
                # 페이지 텍스트를 chunk_length 길이의 청크로 나누어 리스트에 추가(줄바꿈 문자 제거)
        response = client.embeddings.create(model='text-embedding-ada-002', input=chunks) # 임베딩 생성
        print(f"Generated {len(response.data)} embeddings.")
        return [{'id': value.index, 'vector': value.embedding, 'text': chunks[value.index]} for value in response.data]
        # id:청크의 번호, vector:임베딩, text:원본 텍스트 청크

    # Redis에서 검색을 수행하여 텍스트와 벡터 점수를 반환
    # user_query : 검색 쿼리, index_name : 검색할 redisearch 인덱스의 이름,
    # vector_field : 검색할 벡터 필드명 지정, return_fields : 검색 결과에서 반환할 필드 목록
    def search_redis(self,
                     user_query: str, 
                     index_name: str = INDEX_NAME, 
                     text_field : str = "text",
                     vector_field: str = "vector",
                     return_fields: list = ["text", "vector_score"],
                     hybrid_fields="*",
                     k: int = 5,
                     ):
        print("사용자 질문의 키워드 : ",user_query)
        embedded_query = client.embeddings.create(input=user_query,model="text-embedding-ada-002").data[0].embedding
        # hybrid search는 해당 프로젝트 특성 상, 적합하지 않은 것으로 판단
        # keywords = user_query.split(',')
        # keywords = [f"%{keyword.strip()}%" for keyword in keywords]
        # keyword = '|'.join(keywords)
        # base_query = f'(@{text_field}:{keyword})=>[KNN {k} @{vector_field} $vector AS vector_score]'
        base_query = f'(*)=>[KNN {k} @{vector_field} $vector AS vector_score]'
        print("생성된 쿼리:", base_query)  # 쿼리를 출력해 오류를 확인
        query = (
            Query(base_query)
            .return_fields(*return_fields)
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        params_dict = {"vector": np.array(
            embedded_query).astype(dtype=np.float32).tobytes()}
        print("params_dict:", params_dict)  # 변환된 임베딩 확인
        try:
            results = self.redis_client.ft(index_name).search(query, params_dict)
            print(f"Search returned {len(results.docs)} results.")
            if results:
                for i, doc in enumerate(results.docs):
                    score = 1 - float(doc.vector_score)
                    print(f"{i} => {doc.text} (Score: {round(score ,3) })")
            return [doc['text'] for doc in results.docs]
        except Exception as e:
            print(f"Error during search: {e}")  # 오류 발생 시 출력
            return []