import pandas as pd
import time

# --- 설정 ---
# 1. 1:1 매칭이 확인된 원본 영화 파일 (12GB 파일)
original_file_path = 'C:/Users/82109/Downloads/movie_tot.csv' # <--- 원본 파일 경로

# 2. '가중 CBF' 학습용으로 새로 저장할 파일 이름
output_cbf_file = 'C:/Users/82109/Downloads/movies_for_content.csv' 

CHUNK_SIZE = 500000 
# ----------------

print(f"--- 영화 '가중 콘텐츠'(CBF)용 데이터 추출 시작 ---")
print(f"원본: {original_file_path}")
print(f"저장: {output_cbf_file}")
print("파일을 읽고 중복을 제거하는 중입니다...")

start_time = time.time()
chunk_num = 0

columns_to_keep = ['movieId', 'title', 'tag', 'relevance'] 
# --------------------

# 중복 제거된 청크를 저장할 리스트
unique_chunks = []

try:
    with pd.read_csv(original_file_path, sep=',', chunksize=CHUNK_SIZE, usecols=columns_to_keep) as reader:
        for chunk in reader:
            chunk_num += 1
            print(f"  > Processing chunk {chunk_num}...")
            
            # 1. 현재 청크 내에서 'movieId' 기준으로 중복 제거
            unique_chunk = chunk.drop_duplicates(subset=['movieId'])
            unique_chunks.append(unique_chunk)

    # 2. 모든 청크를 하나로 합침
    print("  > 모든 청크를 하나로 합치는 중...")
    df_all_movies = pd.concat(unique_chunks)
    
    # 3. 청크 경계에서 발생할 수 있는 중복을 제거하기 위해 최종 중복 제거
    print("  > 최종 중복 제거 중...")
    df_final_movies = df_all_movies.drop_duplicates(subset=['movieId']).reset_index(drop=True)
    
    # 4. 최종 파일로 저장
    df_final_movies.to_csv(output_cbf_file, index=False)
    
    end_time = time.time()
    
    print("\n=================================================")
    print(f"🎉 CBF용 '가중' 영화 데이터 추출 완료! (총 {end_time - start_time:.2f} 초 소요)")
    print(f"  -> 저장된 파일: {output_cbf_file}")
    print(f"  -> 총 {len(df_final_movies):,} 편의 고유한 영화 정보 저장 완료.")
    print("=================================================")

except Exception as e:
    print(f"\n파일 처리 중 심각한 오류 발생: {e}")