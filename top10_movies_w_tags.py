import pandas as pd
import time

# --- [ 1. "부품" 로드하기 ] ---
print("--- [ 1/2 ] 데이터 로드 중... ---")
start_time = time.time()

# --- 파일 경로 설정 ---
path_prefix = "C:/Users/82109/Downloads/"
movie_ratings_path = path_prefix + "ratings_for_cf.csv"
movie_meta_path = path_prefix + "movies_for_content.csv" 
# --- [새로운 부분] CSV 출력 경로 ---
output_csv_path = path_prefix + "user1_top_movies.csv"

try:
    df_movie_ratings = pd.read_csv(movie_ratings_path)
    df_movie_meta = pd.read_csv(movie_meta_path)
except FileNotFoundError as e:
    print(f"!!! 파일 로드 오류: {e}")
    exit()

print(f"--- [ 2/2 ] 데이터 로드 완료! (소요 시간: {time.time() - start_time:.2f} 초) ---")

# --- [ 3. Top 20 영화 출력 및 CSV 저장 함수 ] ---

def save_user_top_rated_movies(user_id, top_n=20):
    print(f"\n=================================================")
    print(f"🚀 User {user_id} 님을 위한 'Top {top_n} 평가 영화' 목록 저장")
    
    # 1. 'user_id'가 평가한 모든 영화 찾기
    user_ratings = df_movie_ratings[df_movie_ratings['userId'] == user_id]
    if user_ratings.empty:
        print(f"  -> User {user_id} 님의 평점 데이터를 찾을 수 없습니다.")
        return

    # 2. 'rating' 기준으로 내림차순 정렬 후, Top N개 선택
    top_rated_df = user_ratings.sort_values(by='rating', ascending=False).head(top_n)
    
    # 3. 'title'과 'tag' 정보를 'movieId' 기준으로 합치기
    top_movies_details = pd.merge(
        top_rated_df, 
        df_movie_meta[['movieId', 'title', 'tag']], 
        on='movieId', 
        how='left'
    )
    
    # 4. 최종 DataFrame 생성
    final_output_df = top_movies_details[['title', 'tag', 'rating']]
    
    # --- [새로운 부분] CSV 파일로 저장 ---
    try:
        # encoding='utf-8-sig'는 Excel에서 한글이 깨지지 않게 해줍니다.
        final_output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"--- 🏆 Top {top_n} 영화 목록을 {output_csv_path} 에 저장했습니다.")
        
        # 콘솔에도 확인용으로 출력
        with pd.option_context('display.max_colwidth', 70):
            print(final_output_df.to_string(index=False))
            
    except Exception as e:
        print(f"!!! CSV 저장 오류: {e}")
    
    print("=================================================")
    return

# --- [ 4. 함수 실행! ] ---
save_user_top_rated_movies(user_id=1, top_n=20)