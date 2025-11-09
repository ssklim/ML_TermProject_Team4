import pandas as pd
import time

# 1. 파일 경로를 지정해주세요.
file_path = 'C:/Users/82109/Downloads/movie_tot.csv'

CHUNK_SIZE = 500000 

#------------------------------------------------------------------

print(f"--- 12GB 파일 전체 통계 분석 시작 (Chunk size: {CHUNK_SIZE}) ---")
print("파일을 읽는 중입니다. 몇 분 정도 소요될 수 있습니다...")

# 통계를 저장할 변수 초기화
start_time = time.time()
total_rows = 0
unique_users = set()
unique_movies = set()

# 평점 통계
rating_sum = 0
rating_min = float('inf')
rating_max = float('-inf')

# 연도 통계
year_min = float('inf')
year_max = float('-inf')

# 'tag'와 'relevance' 리스트 길이 불일치 카운트
mismatch_count = 0
chunk_num = 0

try:
    # 파일을 청크 단위로 순회
    with pd.read_csv(file_path, sep=',', chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            chunk_num += 1
            print(f"  > Processing chunk {chunk_num}...")
            
            # 1. 총 행 수
            total_rows += len(chunk)
            
            # 2. 고유 사용자 및 영화 ID
            unique_users.update(chunk['userId'])
            unique_movies.update(chunk['movieId'])
            
            # 3. 평점 통계
            rating_sum += chunk['rating'].sum()
            rating_min = min(rating_min, chunk['rating'].min())
            rating_max = max(rating_max, chunk['rating'].max())
            
            # 4. 연도 통계
            year_min = min(year_min, chunk['years'].min())
            year_max = max(year_max, chunk['years'].max())

            # 5. [데이터 무결성 검사] tag와 relevance의 | 개수 일치 확인
            # .str.split('|').str.len()은 각 행의 | 개수를 셉니다.
            try:
                tag_len = chunk['tag'].str.split('|').str.len()
                rel_len = chunk['relevance'].str.split('|').str.len()
                mismatch_count += (tag_len != rel_len).sum()
            except Exception as e:
                print(f"    ! Chunk {chunk_num}에서 'tag'/'relevance' 처리 중 오류: {e}")

    end_time = time.time()
    
    # --- 최종 결과 출력 ---
    print("\n=================================================")
    print("           📊 12GB 데이터 탐색 결과 📊")
    print("=================================================")
    print(f"총 처리 시간: {end_time - start_time:.2f} 초")
    print(f"총 데이터 행 (평점) 수: {total_rows:,}")
    print(f"고유 사용자(userId) 수: {len(unique_users):,}")
    print(f"고유 영화(movieId) 수: {len(unique_movies):,}")
    print("---")
    print("평점(rating) 통계:")
    print(f"  - 평균: {rating_sum / total_rows:.2f}")
    print(f"  - 최소: {rating_min}")
    print(f"  - 최대: {rating_max}")
    print("---")
    print("연도(years) 통계:")
    print(f"  - 최소: {year_min}")
    print(f"  - 최대: {year_max}")
    print("---")
    print("데이터 무결성 검사:")
    print(f"  - 'tag'와 'relevance' 개수 불일치 건수: {mismatch_count}")

except Exception as e:
    print(f"\n파일 처리 중 심각한 오류 발생: {e}")


# print(f"--- 'tag'/'relevance' 불일치 원인 분석 시작 ---")
# print("불일치 데이터를 찾을 때까지 파일을 읽습니다...")

# mismatch_found_count = 0
# chunk_num = 0

# try:
#     with pd.read_csv(file_path, sep=',', chunksize=CHUNK_SIZE) as reader:
#         for chunk in reader:
#             chunk_num += 1
#             print(f"  > Processing chunk {chunk_num}...")
            
#             # 1. NaN(빈 값)이 있으면 str.split()에서 오류가 나므로, 빈 문자열('')로 대체
#             #    (NaN 자체를 '불일치'로 카운트하기 위함)
#             tags_filled = chunk['tag'].fillna('')
#             rels_filled = chunk['relevance'].fillna('')
            
#             # 2. 각 행의 | 개수 계산
#             tag_len = tags_filled.str.split('|').str.len()
#             rel_len = rels_filled.str.split('|').str.len()
            
#             # 3. 불일치하는 행만 필터링
#             mismatched_rows = chunk[tag_len != rel_len]
            
#             if not mismatched_rows.empty:
#                 print(f"\n--- 🚨 불일치 데이터 발견! (Chunk {chunk_num}) ---")
                
#                 # 4. 불일치 샘플 출력
#                 for index, row in mismatched_rows.head(5 - mismatch_found_count).iterrows():
#                     print(f"\n[샘플 {mismatch_found_count + 1}] (Index: {index})")
#                     print(f"  - TAGS (개수: {tag_len.loc[index]}):")
#                     print(f"    {row['tag']}")
#                     print(f"  - RELEVANCE (개수: {rel_len.loc[index]}):")
#                     print(f"    {row['relevance']}")
#                     mismatch_found_count += 1
                
#                 if mismatch_found_count >= 5:
#                     print("\n--- 5개 샘플을 모두 찾았으므로 탐색을 중단합니다. ---")
#                     break # 5개 다 찾았으면 전체 루프 종료

# except Exception as e:
#     print(f"\n파일 처리 중 심각한 오류 발생: {e}")

# if mismatch_found_count == 0:
#     print("\n--- 파일 전체를 스캔했으나 (fillna 처리 후) 불일치 데이터를 찾지 못했습니다. ---")

