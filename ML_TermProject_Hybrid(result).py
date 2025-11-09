import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from surprise import dump, Reader, Dataset
import joblib
import time

print("--- [ 1/5 ] 모든 부품(모델/벡터/데이터) 로드 중... ---")
start_time = time.time()

# 파일 경로 설정
path_prefix = "C:/Users/82109/Downloads/"
svd_model_path = path_prefix + "svd_model.dump"
translator_model_path = path_prefix + "VectorTranslator_model.joblib"
book_vectors_path = path_prefix + "book_vectors.npz"
book_ids_path = path_prefix + "book_id_mapping.csv"
movie_ratings_path = path_prefix + "ratings_for_cf.csv"
book_meta_path = path_prefix + "books_for_content.csv"

try:
    # 1. SVD 모델 (사용자 잠재 벡터용)
    _, svd_model = dump.load(svd_model_path)
    
    # 2. "번역 모델"
    translator_model = joblib.load(translator_model_path)
    
    # 3. 책 콘텐츠 벡터 (번역할 대상)
    book_vectors = load_npz(book_vectors_path)
    
    # 4. ID 매핑 및 기타 데이터
    df_book_ids = pd.read_csv(book_ids_path)
    df_movie_ratings = pd.read_csv(movie_ratings_path)
    df_book_meta = pd.read_csv(book_meta_path).set_index('book_id')

except FileNotFoundError as e:
    print(f"!!! 파일 로드 오류: {e}")
    exit()

# 빠른 조회를 위한 "매핑" 생성
print("--- [ 2/5 ] SVD 사용자 매핑(Dictionary) 생성 중... ---")
# SVD 모델의 내부 ID(inner_id)와 실제 ID(raw_id) 매핑
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df_movie_ratings, reader)
trainset = data.build_full_trainset()
user_raw_to_inner = {trainset.to_raw_uid(inner_id): inner_id for inner_id in trainset.all_users()}
book_index_to_id = {index: book_id for index, book_id in enumerate(df_book_ids['book_id'])}

# SVD가 학습한 실제 '사용자 잠재 벡터' (100차원)
user_latent_vectors = svd_model.pu

# (9천 권의 책 콘텐츠 벡터를 "추정된 잠재 벡터"로 미리 번역) 
# book_vectors (9759, 5000) -> translator -> (9759, 100)
print("--- [ 3/5 ] '번역 모델'로 모든 책 벡터 번역 중... (시간 소요) ---")
estimated_book_latent_vectors = translator_model.predict(book_vectors)
print("  > 모든 책 '잠재 벡터' 번역 완료.")

print(f"--- [ 4/5 ] 부품 로드 및 준비 완료! (소요 시간: {time.time() - start_time:.2f} 초) ---")

# 핵심 추천 함수 정의 (SVD 공간)

def get_hybrid_recommendations(user_id, top_n=10):
    """
    SVD 잠재 공간에서 사용자의 취향과 "번역된" 책 벡터를 비교하여 추천합니다.
    """
    print(f"\n=================================================")
    print(f" User {user_id} 님을 위한 '하이브리드' 책 추천 시작...")
    
    # 1. 'user_id'의 "사용자 잠재 벡터" (100차원) 가져오기
    inner_uid = user_raw_to_inner.get(user_id)
    if inner_uid is None:
        print(f"  -> User {user_id} 님은 SVD 모델이 모르는 사용자입니다 (영화 평점 없음).")
        return None
        
    user_vector = user_latent_vectors[inner_uid] # (100,)
    
    # 2. (매칭) "사용자 벡터"와 "번역된 책 벡터" 9천 개 내적(Dot Product)
    # user_vector (100,) 와 estimated_book_latent_vectors (9759, 100)

    scores = user_vector.dot(estimated_book_latent_vectors.T) # (9759,)
    
    # 3. 취향 일치 점수가 높은 순서대로 정렬
    # (점수, 책의 vector_index) 튜플 리스트 생성
    book_scores = list(enumerate(scores))
    
    # 점수(x[1])를 기준으로 내림차순 정렬
    sorted_book_scores = sorted(book_scores, key=lambda x: x[1], reverse=True)
    
    # 4. 최종 추천 목록 생성
    print(f"---  User {user_id} 님을 위한 Top {top_n} 추천 도서 ---")
    rec_count = 0
    for book_index, score in sorted_book_scores:
        if rec_count >= top_n:
            break
        
        # vector_index를 실제 book_id로 변환
        book_id = book_index_to_id.get(book_index)
        if book_id:
            try:
                # book_id로 책 제목 찾기
                title = df_book_meta.loc[book_id]['title']
                print(f"  {rec_count+1}. {title} (취향 일치 점수: {score:.4f})")
                rec_count += 1
            except KeyError:
                pass
                
    print("=================================================")
    return

# 추천 시스템 실행
get_hybrid_recommendations(user_id=1, top_n=10)