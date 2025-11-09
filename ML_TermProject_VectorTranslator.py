import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from surprise import dump, Reader, Dataset
import joblib
from sklearn.neighbors import KNeighborsRegressor
import time

# [ 1. 부품 준비 ]
print("---  '번역 모델' 훈련 시작 ---")
print("[ 1/4 ] SVD모델, TF-IDF벡터, 평점데이터 로드 중...")
start_time = time.time()

# --- 파일 경로 설정 (본인 환경에 맞게 확인!) ---
path_prefix = "C:/Users/82109/Downloads/"
svd_model_path = path_prefix + "svd_model.dump"
movie_vectors_path = path_prefix + "movie_vectors.npz"
movie_ids_path = path_prefix + "movie_id_mapping.csv"
movie_ratings_path = path_prefix + "ratings_for_cf.csv"

# 2. 훈련된 '번역 모델'을 저장할 경로
output_translator_model = path_prefix + "VectorTranslator_model.joblib"

try:
    # 1. SVD 모델 로드
    _, svd_model = dump.load(svd_model_path)
    
    # 2. TF-IDF 영화 콘텐츠 벡터 로드
    movie_vectors = load_npz(movie_vectors_path)
    
    # 3. ID 매핑 및 평점 데이터 로드
    df_movie_ids = pd.read_csv(movie_ids_path)
    df_movie_ratings = pd.read_csv(movie_ratings_path)

except FileNotFoundError as e:
    print(f"!!! 파일 로드 오류: {e}")
    exit()

# [ 2. '번역' 훈련용 X, Y 데이터셋 구축 ]
print("[ 2/4 ] '번역' 훈련용 X(콘텐츠), Y(잠재) 데이터셋 구축 중...")

# 1. SVD 모델의 내부 ID(inner_id)와 실제 ID(raw_id) 매핑
# SVD 모델을 훈련시켰던 데이터 로드
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df_movie_ratings, reader)
trainset = data.build_full_trainset()
# SVD가 학습한 실제 '영화 잠재 벡터' (100차원)
movie_latent_vectors = svd_model.qi # Y (정답)

# 2. {movieId: vector_index} 매핑 (TF-IDF 벡터용)
movie_id_to_index = {movie_id: index for index, movie_id in enumerate(df_movie_ids['movieId'])}

# 훈련 데이터를 담을 리스트
X_train_list = []
Y_train_list = []

# 3. 1.2만 편의 영화를 순회하며 X, Y 짝짓기
for movie_id in df_movie_ids['movieId']:
    try:
        # Y (정답) 찾기: SVD 모델의 inner_id (내부 ID)
        inner_iid = trainset.to_inner_iid(movie_id)
        y_vec = movie_latent_vectors[inner_iid]
        
        # X (입력) 찾기: TF-IDF 모델의 index (순서)
        vector_index = movie_id_to_index.get(movie_id)
        x_vec = movie_vectors[vector_index]
        
        # 둘 다 찾았으면 리스트에 추가
        X_train_list.append(x_vec)
        Y_train_list.append(y_vec)
        
    except ValueError:
        # trainset.to_inner_iid()가 SVD 훈련셋에 없는 movie_id를 만나면 오류 발생
        # (드물지만 발생 가능, 이 경우 그냥 건너뜀)
        pass

# 4. 리스트를 최종 행렬로 변환
# X는 희소 행렬이므로 vstack 사용
from scipy.sparse import vstack
X_train = vstack(X_train_list)
Y_train = np.array(Y_train_list)

print(f"  > 훈련 데이터 {X_train.shape[0]}개 생성 완료.")
print(f"  > X (콘텐츠) Shape: {X_train.shape}")
print(f"  > Y (잠재 벡터) Shape: {Y_train.shape}")

# [ 3. "번역 모델" 학습 (K-NN) ]
print("[ 3/4 ] K-NN 회귀 모델('번역기') 훈련 시작...")
# n_neighbors=5 : 가장 가까운 5개의 영화 벡터를 참고하여 '번역'
# n_jobs=-1 : CPU 코어를 모두 사용하여 빠르게 학습
translator_model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)

translator_model.fit(X_train, Y_train)

fit_time = time.time()
print(f"  > '번역기' 훈련 완료. (소요 시간: {fit_time - start_time:.2f} 초)")

# [ 4. "번역 모델" 저장 ]
print("[ 4/4 ] 훈련된 '번역 모델' 파일로 저장 중...")
joblib.dump(translator_model, output_translator_model)

end_time = time.time()
print("\n=================================================")
print(f" 3단계 '번역 모델' 생성 완료! (총 {end_time - start_time:.2f} 초)")
print(f"  -> 저장된 파일: {output_translator_model}")
print("=================================================")