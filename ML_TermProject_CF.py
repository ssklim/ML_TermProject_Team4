import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise import dump  # surprise의 저장 기능
import time

# --- 설정 ---
cf_file_path = 'C:/Users/82109/Downloads/ratings_for_cf.csv'

# SVD 모델을 저장할 파일 경로
output_svd_model = 'C:/Users/82109/Downloads/svd_model.dump'

print("--- 1단계: SVD 모델 '최종 학습' 및 '저장' 시작 ---")

# 1. Surprise용 데이터 로드
start_time = time.time()
print("데이터를 로드하는 중입니다...")
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(pd.read_csv(cf_file_path), reader)

# 2. '전체 데이터'를 훈련셋으로 사용
print("전체 데이터를 훈련셋으로 변환하는 중...")
trainset = data.build_full_trainset()

load_time = time.time()
print(f"데이터 준비 완료. (소요 시간: {load_time - start_time:.2f} 초)")

# 3. SVD 모델 초기화 및 학습
print("SVD 모델 학습 시작 (전체 데이터)...")
model = SVD(n_factors=100, n_epochs=20, random_state=42, verbose=True)

model.fit(trainset)

fit_time = time.time()
print(f"모델 학습 완료. (소요 시간: {fit_time - load_time:.2f} 초)")

# 4. 훈련된 SVD 모델 파일로 저장
print("SVD 모델 파일로 저장 중...")
dump.dump(output_svd_model, algo=model)
print(f"모델 저장 완료: {output_svd_model}")
# --------------------

end_time = time.time()
print("\n=================================================")
print(f" 1단계 SVD 모델 저장 완료 (총 소요 시간: {end_time - start_time:.2f} 초)")
print(f"  -> 저장된 파일: {output_svd_model}")
print("=================================================")