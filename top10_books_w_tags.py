import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from surprise import dump, Reader, Dataset
import joblib
import time

# --- [ 1. "부품" 로드하기 (동일) ] ---
print("--- [ 1/5 ] 모든 부품(모델/벡터/데이터) 로드 중... ---")
start_time = time.time()
path_prefix = "C:/Users/82109/Downloads/"
svd_model_path = path_prefix + "svd_model.dump"
translator_model_path = path_prefix + "VectorTranslator_model.joblib"
book_vectors_path = path_prefix + "book_vectors.npz"
book_ids_path = path_prefix + "book_id_mapping.csv"
movie_ratings_path = path_prefix + "ratings_for_cf.csv" 
book_meta_path = path_prefix + "books_for_content.csv" 
# --- [새로운 부분] CSV 출력 경로 ---
output_csv_path_recs = path_prefix + "user1_recommended_books.csv"

# (try-except 블록은 동일합니다... 생략)
try:
    _, svd_model = dump.load(svd_model_path)
    translator_model = joblib.load(translator_model_path)
    book_vectors = load_npz(book_vectors_path)
    df_book_ids = pd.read_csv(book_ids_path)
    df_movie_ratings = pd.read_csv(movie_ratings_path)
    df_book_meta = pd.read_csv(book_meta_path).set_index('book_id')
except FileNotFoundError as e:
    print(f"!!! 파일 로드 오류: {e}")
    exit()

# --- [ 2. 매핑 생성 (동일) ] ---
print("--- [ 2/5 ] SVD 사용자 매핑(Dictionary) 생성 중... ---")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df_movie_ratings, reader)
trainset = data.build_full_trainset()
user_raw_to_inner = {trainset.to_raw_uid(inner_id): inner_id for inner_id in trainset.all_users()}
book_index_to_id = {index: book_id for index, book_id in enumerate(df_book_ids['book_id'])}
user_latent_vectors = svd_model.pu

# --- [ 3. "책 번역" (동일) ] ---
print("--- [ 3/5 ] '번역 모델'로 모든 책 벡터 번역 중... ---")
estimated_book_latent_vectors = translator_model.predict(book_vectors)
print(f"--- [ 4/5 ] 부품 로드 및 준비 완료! (소요 시간: {time.time() - start_time:.2f} 초) ---")

# --- [ 5. 핵심 추천 함수 정의 (CSV 저장 기능 추가) ] ---

def save_hybrid_recommendations(user_id, top_n=10):
    print(f"\n=================================================")
    print(f"🚀 User {user_id} 님을 위한 '하이브리드' 책 추천 저장...")
    
    inner_uid = user_raw_to_inner.get(user_id)
    if inner_uid is None:
        print(f"  -> User {user_id} 님은 SVD 모델이 모르는 사용자입니다.")
        return None
        
    user_vector = user_latent_vectors[inner_uid] 
    scores = user_vector.dot(estimated_book_latent_vectors.T) 
    book_scores = list(enumerate(scores))
    sorted_book_scores = sorted(book_scores, key=lambda x: x[1], reverse=True)
    
    # --- [새로운 부분] 결과를 리스트에 저장 ---
    recommendations_list = []
    rec_count = 0
    
    for book_index, score in sorted_book_scores:
        if rec_count >= top_n:
            break
        
        book_id = book_index_to_id.get(book_index)
        if book_id:
            try:
                book_data = df_book_meta.loc[book_id]
                title = book_data['title']
                tags = book_data['tag_list']
                
                # 리스트에 딕셔너리 형태로 추가
                recommendations_list.append({
                    'rank': rec_count + 1,
                    'title': title,
                    'tags': tags,
                    'match_score': score
                })
                rec_count += 1
            except KeyError:
                pass
    
    # --- [새로운 부분] 리스트를 DataFrame으로 변환 후 CSV 저장 ---
    if recommendations_list:
        recs_df = pd.DataFrame(recommendations_list)
        try:
            recs_df.to_csv(output_csv_path_recs, index=False, encoding='utf-8-sig')
            print(f"--- 🏆 Top {top_n} 추천 도서 목록을 {output_csv_path_recs} 에 저장했습니다.")
            
            # 콘솔에도 확인용으로 출력
            with pd.option_context('display.max_colwidth', 70):
                print(recs_df.to_string(index=False))
                
        except Exception as e:
            print(f"!!! CSV 저장 오류: {e}")
    else:
        print("--- 추천 목록을 생성하지 못했습니다. ---")
                
    print("=================================================")
    return

# --- [ 6. 함수 실행! ] ---
save_hybrid_recommendations(user_id=1, top_n=10)