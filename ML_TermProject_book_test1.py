import pandas as pd
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

file_path = 'C:/Users/82109/Downloads/goodbooks/content/goodbooks.csv'

print(f"--- 전처리된 책(Book) 데이터 탐색 시작 ---")
print(f"파일: {file_path}")

# --- 1. 파일 탐색 ---
try:
    df_books = pd.read_csv(file_path, nrows=5)
    
    print("\n--- 샘플 데이터 (head) ---")
    pd.set_option('display.max_columns', None) # 모든 컬럼 표시
    print(df_books)
    
    print("\n--- 데이터 정보 (info) ---")
    df_books.info()

except Exception as e:
    print(f"\n파일 읽기 오류: {e}")

print("\n--- 탐색 완료 ---")