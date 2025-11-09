import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import joblib  # for saving models
from scipy.sparse import save_npz # for saving vectors
import numpy as np
import re

# --- 설정 ---
# 1. 입력 파일 경로
movie_content_file = 'C:/Users/82109/Downloads/movies_for_content.csv'
book_content_file = 'C:/Users/82109/Downloads/books_for_content.csv'

# 2. 출력 (저장) 파일 경로
output_tfidf_model = 'C:/Users/82109/Downloads/tfidf_model.joblib'
output_movie_vectors = 'C:/Users/82109/Downloads/movie_vectors.npz'
output_book_vectors = 'C:/Users/82109/Downloads/book_vectors.npz'
output_movie_ids = 'C:/Users/82109/Downloads/movie_id_mapping.csv'
output_book_ids = 'C:/Users/82109/Downloads/book_id_mapping.csv'
# ----------------

print("--- 2단계: '가중 TF-IDF' 모델 훈련 시작 ---")
start_time = time.time()

# 2-A: 영화 '가중 문서' 생성 함수
def create_weighted_movie_doc(row):
    try:
        # 1. title을 소문자로
        title = str(row['title']).lower()
        
        # 2. tag와 relevance를 리스트로 분리
        tags = str(row['tag']).split('|')
        relevances = str(row['relevance']).split('|')
        
        doc_parts = [title]
        
        # 3. 1:1 매칭이 확인되었으므로, relevance 점수를 "반복 횟수"로 변환
        if len(tags) == len(relevances):
            for tag, rel_str in zip(tags, relevances):
                # 태그 처리: 소문자로, 공백 제거 (예: 'Black comedy' -> 'blackcomedy')
                processed_tag = re.sub(r'\s+', '', tag.lower())
                
                # relevance 점수 (1.0 ~ 0.0) -> 반복 횟수 (10 ~ 0)
                repeat_count = int(float(rel_str) * 10) 
                
                # 점수 0.1(1번 반복) 이상인 태그만 문서에 추가
                if repeat_count > 0:
                    # 'processed_tag'를 repeat_count 만큼 반복해서 리스트에 추가
                    tag_repeated = ' '.join([processed_tag] * repeat_count)
                    doc_parts.append(tag_repeated)
        
        return ' '.join(doc_parts)
    except Exception as e:
        print(f"Error processing movie row {row['movieId']}: {e}")
        return "" # 오류 발생 시 빈 문서 반환

# 2-B: 책 '일반 문서' 생성 함수
def create_book_doc(row):
    try:
        # 1. title을 소문자로
        title = str(row['title']).lower()
        
        # 2. tag_list (NaN 처리)를 소문자로, 콤마(,)를 공백(' ')으로 변경
        # (예: 'to-read,fiction' -> 'to-read fiction')
        tags = str(row['tag_list']).lower().replace(',', ' ')
        # 태그 내의 '-'는 의미가 있을 수 있으므로 유지 (예: 'to-read')
        
        return title + ' ' + tags
    except Exception as e:
        print(f"Error processing book row {row['book_id']}: {e}")
        return ""

# 파일 로드
print("콘텐츠 파일 로드 중...")
df_movies = pd.read_csv(movie_content_file)
df_books = pd.read_csv(book_content_file)

# '문서' 컬럼 생성
print("영화 '가중 문서' 생성 중...")
df_movies['document'] = df_movies.apply(create_weighted_movie_doc, axis=1)

print("책 '일반 문서' 생성 중...")
df_books['document'] = df_books.apply(create_book_doc, axis=1)

# TF-IDF 훈련
print("영화/책 문서를 합쳐 TF-IDF 모델 훈련 시작...")

# 3-A. 훈련시킬 전체 말뭉치(corpus) 생성
corpus = pd.concat([df_movies['document'], df_books['document']])

# 3-B. TF-IDF 모델 초기화
# max_features=5000 : 가장 중요한 5000개 태그(단어)만 벡터로 사용
# stop_words='english' : 'a', 'the', 'is' 등 의미 없는 영단어 제거
tfidf_model = TfidfVectorizer(max_features=5000, stop_words='english')

# 3-C. 전체 말뭉치로 모델 훈련 (fit)
tfidf_model.fit(corpus)
print("TF-IDF 모델 훈련 완료.")

# 4. 벡터 변환 및 저장
print("훈련된 모델로 영화/책 문서를 벡터로 변환 중...")
# 4-A. 영화 문서를 '콘텐츠 벡터'로 변환 (transform)
movie_vectors = tfidf_model.transform(df_movies['document'])
# 4-B. 책 문서를 '콘텐츠 벡터'로 변환 (transform)
book_vectors = tfidf_model.transform(df_books['document'])

print("모델 및 벡터 파일 저장 중...")
# 4-C. 훈련된 TF-IDF 모델 저장
joblib.dump(tfidf_model, output_tfidf_model)

# 4-D. 생성된 벡터 저장 (npz 형식)
save_npz(output_movie_vectors, movie_vectors)
save_npz(output_book_vectors, book_vectors)

# 4-E. 벡터의 순서가 ID와 일치하도록 ID 리스트 저장
df_movies[['movieId']].to_csv(output_movie_ids, index=False)
df_books[['book_id']].to_csv(output_book_ids, index=False)

end_time = time.time()
print("\n=================================================")
print(f" 2단계 콘텐츠 벡터 생성 완료 (총 {end_time - start_time:.2f} 초 소요)")
print(f"  -> TF-IDF 모델 저장: {output_tfidf_model}")
print(f"  -> 영화 벡터 (Shape: {movie_vectors.shape}) 저장: {output_movie_vectors}")
print(f"  -> 책 벡터 (Shape: {book_vectors.shape}) 저장: {output_book_vectors}")
print("=================================================")