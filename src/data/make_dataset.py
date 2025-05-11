from pathlib import Path
import pandas as pd
import numpy as np

def main():
    # 1) 파일 읽기
    df = pd.read_csv('data/raw/lendingclub.csv', low_memory = False) 
    print(df.head(3))
   
    
    def process_emp_length(x):
        if pd.isna(x):
            return None
        elif '< 1' in x:
            return 0.5
        elif '10+' in x:
            return 10.0
        else:
            extracted = pd.to_numeric(pd.Series(x).str.extract(r'(\d+)')[0], errors='coerce')
            return extracted.iloc[0]

    df['emp_length'] = df['emp_length'].apply(process_emp_length)
    df= df.drop('emp_title', axis =1)

    E_map = {'Charged Off':1, 'Default':1}
    df['E'] = df['loan_status'].map(E_map).fillna(0).astype(int)

    # ================== Survival-analysis specific preprocessing ==================
    # 날짜형 변환 (Lending-Club 날짜 포맷: 'Dec-2015' → '%b-%Y')
    df['issue_d']      = pd.to_datetime(df['issue_d'], errors='coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], errors='coerce')
    df = df[df['issue_d'] >= '2019-01-01']
    
    # 관측 종료 시점(obs_end) 설정: Current → cutoff, 그 외 → last_pymnt_d
    cutoff_date = pd.to_datetime('2020-12-31')
    df['obs_end'] = df['last_pymnt_d'].where(df['E'] == 1, cutoff_date)

    df['T'] = (df['obs_end'].dt.year - df['issue_d'].dt.year) * 12 + (df['obs_end'].dt.month - df['issue_d'].dt.month)
    df = df[df['T'].notna()]
    df['T'] = df['T'].clip(lower=0)

    print(f"전처리 후 데이터 크기: {df.shape}")

    # 3) inim 폴더에 저장
    out_path = Path('data/interim/lendingclub_survival.csv')
    out_path.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(out_path, index = False)
    print(f"✅ 저장 완료: {out_path}")

if __name__ == '__main__':
    main()