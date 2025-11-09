import pandas as pd
import time


original_book_file = 'C:/Users/82109/Downloads/goodbooks/content/goodbooks.csv'

output_cbf_file = 'C:/Users/82109/Downloads/books_for_content.csv'

# 파일 청크 처리
CHUNK_SIZE = 500000 
# ----------------

print(f"--- 책 콘텐츠(CBF)용 데이터 추출 시작 ---")
print(f"원본: {original_book_file}")
print(f"저장: {output_cbf_file}")
print("파일을 읽고 중복을 제거하는 중입니다...")

start_time = time.time()
chunk_num = 0

# 필요한 컬럼만 지정
columns_to_keep = ['book_id', 'title', 'tag_list']

# 중복 제거된 청크를 저장할 리스트
unique_chunks = []

try:
    with pd.read_csv(original_book_file, sep=',', chunksize=CHUNK_SIZE, usecols=columns_to_keep) as reader:
        for chunk in reader:
            chunk_num += 1
            print(f"  > Processing chunk {chunk_num}...")
            
            # 1. 현재 청크 내에서 'book_id' 기준으로 중복 제거
            unique_chunk = chunk.drop_duplicates(subset=['book_id'])
            unique_chunks.append(unique_chunk)

    # 2. 모든 청크를 하나로 합침
    print("  > 모든 청크를 하나로 합치는 중...")
    df_all_books = pd.concat(unique_chunks)
    
    # 3. 청크 경계에서 발생할 수 있는 중복을 제거하기 위해 최종 중복 제거
    print("  > 최종 중복 제거 중...")
    df_final_books = df_all_books.drop_duplicates(subset=['book_id']).reset_index(drop=True)
    
    # 4. 최종 파일로 저장
    df_final_books.to_csv(output_cbf_file, index=False)
    
    end_time = time.time()
    
    print("\n=================================================")
    print(f"🎉 CBF용 책 데이터 추출 완료! (총 {end_time - start_time:.2f} 초 소요)")
    print(f"  -> 저장된 파일: {output_cbf_file}")
    print(f"  -> 총 {len(df_final_books):,} 권의 고유한 책 정보 저장 완료.")
    print("=================================================")

except Exception as e:
    print(f"\n파일 처리 중 심각한 오류 발생: {e}")