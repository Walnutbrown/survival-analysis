import pandas as pd
from pandas.api import types as ptypes
import numpy as np

def main():
    # 1) interim 데이터 불러오기
    df = pd.read_csv('../../data/interim/lendingclub_survival.csv')

    # 2) 숫자형으로 변환해야 하는 컬럼 먼저 처리
    df['term'] = df['term'].str.extract(r'(\d+)').astype(float)

    # 퍼센트(%) 기호가 들어간 컬럼 자동 탐색 및 처리
    object_cols = df.select_dtypes(include='object').columns.tolist()

    percent_cols = []
    for col in object_cols:
        if df[col].dropna().apply(lambda x: isinstance(x, str) and '%' in x).any():
            percent_cols.append(col)

    for col in percent_cols:
        df[col] = df[col].str.replace('%', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 퍼센트 기호는 없지만 숫자로 변환해야 하는 컬럼 처리
    plain_percent_cols = ['percent_bc_gt_75', 'all_util', 'il_util']
    for col in plain_percent_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 날짜형 변환
    df['issue_d'] = pd.to_datetime(df['issue_d'], errors = 'coerce')
    df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], errors = 'coerce')
    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors = 'coerce')
    df['sec_app_earliest_cr_line'] = pd.to_datetime(df['sec_app_earliest_cr_line'], errors = 'coerce')
    df['obs_end'] = pd.to_datetime(df['obs_end'], errors = 'coerce')

    # earliest_cr_line과 sec_app_earliest_cr_line 개월수 계산
    df['earliest_cr_line_num'] = (df['earliest_cr_line'].dt.year - df['issue_d'].dt.year) * 12 + (df['earliest_cr_line'].dt.month - df['issue_d'].dt.month)
    df['sec_app_earliest_cr_line_num'] = (df['sec_app_earliest_cr_line'].dt.year - df['issue_d'].dt.year) * 12 + (df['sec_app_earliest_cr_line'].dt.month - df['issue_d'].dt.month)
    df.drop(columns=['earliest_cr_line', 'sec_app_earliest_cr_line'], inplace=True)

    # 3) Feature Engineering
    features = [
    'acc_now_delinq', 'acc_open_past_24mths', 'addr_state', 'all_util',
    'annual_inc', 'annual_inc_joint', 'application_type', 'avg_cur_bal',
    'bc_open_to_buy', 'bc_util', 'chargeoff_within_12_mths',
    'collections_12_mths_ex_med', 'delinq_2yrs',
    'delinq_amnt', 'dti', 'dti_joint', 'earliest_cr_line_num',
    'emp_length', 'fico_range_high', 'fico_range_low', 'home_ownership',
    'il_util', 'inq_fi', 'inq_last_12m', 'inq_last_6mths', 'max_bal_bc',
    'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op', 'mo_sin_rcnt_rev_tl_op',
    'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_last_delinq',
    'mths_since_last_major_derog', 'mths_since_last_record',
    'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
    'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl',
    'num_bc_sats', 'num_bc_tl', 'num_il_tl', 'num_op_rev_tl',
    'num_rev_accts', 'num_rev_tl_bal_gt_0', 'num_sats',
    'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
    'num_tl_op_past_12m', 'open_acc', 'open_acc_6m', 'open_act_il',
    'open_il_12m', 'open_il_24m', 'open_rv_12m', 'open_rv_24m',
    'pct_tl_nvr_dlq', 'percent_bc_gt_75', 'pub_rec', 'pub_rec_bankruptcies',
    'purpose', 'revol_bal', 'revol_util', 'tax_liens',
    'tot_coll_amt', 'tot_cur_bal', 'tot_hi_cred_lim', 'total_acc',
    'total_bal_ex_mort', 'total_bal_il', 'total_bc_limit', 'total_cu_tl',
    'total_il_high_credit_limit', 'total_rev_hi_lim', 'verification_status',
    'verification_status_joint', 'revol_bal_joint', 'sec_app_fico_range_low',
    'sec_app_fico_range_high', 'sec_app_earliest_cr_line_num',
    'sec_app_inq_last_6mths', 'sec_app_mort_acc', 'sec_app_open_acc',
    'sec_app_revol_util', 'sec_app_open_act_il', 'sec_app_num_rev_accts',
    'sec_app_chargeoff_within_12_mths', 'sec_app_collections_12_mths_ex_med'
    ]

   
    
            
    # Fragmentation 문제 방지
    df = df.copy()

    # 4) 결측치 처리
    for col in features:
        if df[col].isnull().any():
            # Categorical columns: add 'missing' category then fill
            if ptypes.is_categorical_dtype(df[col]):
                df[col] = df[col].cat.add_categories('missing')
                df[col] = df[col].fillna('missing')
            # Numeric columns: indicator + fill 0
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-999)
                df[col+'_missing'] = df[col].isnull().astype(int)
            # All other columns: fill with string 'missing'
            else:
                df[col] = df[col].fillna('missing')
    
    # 5) Label Encoding
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes.replace(-1, -999)

    # 6) 저장
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    df.to_parquet('../../data/processed/lendingclub_features_for_rf.parquet', index=False, engine='pyarrow')
    print('✅ 파일 저장 완료')
    
if __name__ == "__main__":
    main()