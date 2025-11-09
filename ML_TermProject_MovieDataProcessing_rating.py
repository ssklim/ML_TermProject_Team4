import pandas as pd
import time
# --- 설정 ---
# 1. 1:1 매칭이 확인된 원본 파일
original_file_path = 'C:/Users/82109/Downloads/movie_tot.csv'

# 2. CF 학습용으로 새로 저장할 파일 이름
output_cf_file = 'C:/Users/82109/Downloads/ratings_for_cf.csv'

CHUNK_SIZE = 500000 
# ----------------

print(f"--- 협업 필터링(CF)용 데이터 추출 시작 ---")
print(f"원본: {original_file_path}")
print(f"저장: {output_cf_file}")
print("파일을 읽고 저장하는 중입니다. 몇 분 정도 소요될 수 있습니다...")

start_time = time.time()
chunk_num = 0

# 필요한 컬럼만 지정
columns_to_keep = ['userId', 'movieId', 'rating']

try:
    with pd.read_csv(original_file_path, sep=',', chunksize=CHUNK_SIZE, usecols=columns_to_keep) as reader:
        for chunk in reader:
            chunk_num += 1
            print(f"  > Processing chunk {chunk_num}...")
            
            # 1. 첫 번째 청크는 헤더를 포함하여 새로 쓰기
            if chunk_num == 1:
                chunk.to_csv(output_cf_file, mode='w', index=False, header=True)
            # 2. 이후 청크는 헤더 없이 이어서 쓰기
            else:
                chunk.to_csv(output_cf_file, mode='a', index=False, header=False)

    end_time = time.time()
    
    print("\n=================================================")
    print(f"🎉 CF용 데이터 추출 완료! (총 {end_time - start_time:.2f} 초 소요)")
    print(f"  -> 저장된 파일: {output_cf_file}")
    print("=================================================")
    print("\n이제 이 'ratings_for_cf.csv' 파일로 SVD 모델 학습을 시작할 수 있습니다.")

except Exception as e:
    print(f"\n파일 처리 중 심각한 오류 발생: {e}")