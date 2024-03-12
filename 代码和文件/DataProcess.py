'''
【python环境3.9.7】
首先导入xlsx文件，需要先安装环境 pip install openpyx1
导入原始数据，
'''

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

rawpath = "Data/RawData"
files = [f for f in os.listdir(rawpath) if f.endswith('.xlsx')]

######################################## 缺失值统计 ########################################
result_loanData = pd.DataFrame(columns=["文件名", "特征列", "缺失值个数", "缺失值占比"])
for file in files:
    file_path = os.path.join(rawpath, file)
    LoanData = pd.read_excel(file_path)

    missing_values = LoanData.isnull().sum()  
    missing_percentage = (missing_values / len(LoanData)) * 100  #统计缺失值的占比

    for column in LoanData.columns:
        result_loanData = result_loanData.append({"文件名": file, 
                                                  "特征列": column, 
                                                  "缺失值个数": missing_values[column], 
                                                  "缺失值占比(%)": missing_percentage[column]},
                                                  ignore_index = True)

result_loanData.to_excel("Data/MissingData/MissingValuesResult.xlsx", index = False)

######################################## 筛选特征列 ########################################
# 定义保留或修改的列
columns_judege = {"id":"删除", "member_id":"删除",  "loan_amnt":"保留", "funded_amnt":"删除", "funded_amnt_inv":"删除", "term":"保留",
                  "int_rate":"保留", "installment":"保留", "grade":"删除", "sub_grade":"删除", "emp_title":"删除", "emp_length":"保留",
                  "home_ownership":"保留", "annual_inc":"保留", "verification_status":"删除", "issue_d":"删除", "loan_status":"保留",
                  "pymnt_plan":"删除", "url":"删除", "desc":"删除", "purpose":"保留", "title":"保留", "zip_code":"删除", "addr_state":"保留",
                  "dti":"保留", "delinq_2yrs":"保留", "earliest_cr_line":"删除", "inq_last_6mths":"删除", "mths_since_last_delinq":"保留",
                  "mths_since_last_record":"删除" , "open_acc":"保留", "pub_rec":"保留", "revol_bal":"保留", "revol_util":"保留", "total_acc":"保留",
                  "initial_list_status":"删除", "out_prncp":"保留", "out_prncp_inv":"删除", "total_pymnt":"删除", "total_pymnt_inv":"删除",
                  "total_rec_prncp":"删除", "total_rec_int":"删除", "total_rec_late_fee":"删除", "recoveries":"删除", "collection_recovery_fee":"删除",
                  "last_pymnt_d":"删除", "last_pymnt_amnt":"保留", "next_pymnt_d":"删除", "last_credit_pull_d":"删除", "collections_12_mths_ex_med":"删除",
                  "mths_since_last_major_derog":"删除", "policy_code":"删除", "application_type":"删除", "annual_inc_joint":"删除", "dti_joint":"删除",
                  "verification_status_joint":"删除", "acc_now_delinq":"保留", "tot_coll_amt":"保留", "tot_cur_bal":"保留", "open_acc_6m":"保留",
                  "open_act_il":"保留", "open_il_12m":"保留", "open_il_24m":"保留", "mths_since_rcnt_il":"删除", "total_bal_il":"保留", "il_util":"保留",
                  "open_rv_12m":"删除", "open_rv_24m":"删除", "max_bal_bc":"删除", "all_util":"删除", "total_rev_hi_lim":"删除", "inq_fi":"删除",
                  "total_cu_tl":"删除", "inq_last_12m":"删除", "acc_open_past_24mths":"删除", "avg_cur_bal":"保留", "bc_open_to_buy":"删除", "bc_util":"删除",
                  "chargeoff_within_12_mths":"删除", "delinq_amnt":"保留", "mo_sin_old_il_acct":"保留", "mo_sin_old_rev_tl_op":"保留", "mo_sin_rcnt_rev_tl_op":"保留",
                  "mo_sin_rcnt_tl":"保留", "mort_acc":"保留", "mths_since_recent_bc":"保留", "mths_since_recent_bc_dlq":"保留", "mths_since_recent_inq":"删除",
                  "mths_since_recent_revol_delinq":"保留", "num_accts_ever_120_pd":"保留", "num_actv_bc_tl":"保留", "num_actv_rev_tl":"保留", "num_bc_sats":"保留",
                  "num_bc_tl":"保留", "num_il_tl":"保留", "num_op_rev_tl":"保留", "num_rev_accts":"保留", "num_rev_tl_bal_gt_0":"保留", "num_sats":"删除",
                  "num_tl_120dpd_2m":"保留", "num_tl_30dpd":"保留", "num_tl_90g_dpd_24m":"保留", "num_tl_op_past_12m":"删除", "pct_tl_nvr_dlq":"保留",
                  "percent_bc_gt_75":"删除", "pub_rec_bankruptcies":"保留", "tax_liens":"保留", "tot_hi_cred_lim":"保留", "total_bal_ex_mort":"保留","total_bc_limit":"保留",
                  "total_il_high_credit_limit":"保留", "revol_bal_joint":"删除", "sec_app_earliest_cr_line":"删除", "sec_app_inq_last_6mths":"删除", "sec_app_mort_acc":"删除",
                  "sec_app_open_acc":"删除", "sec_app_revol_util":"删除", "sec_app_open_act_il":"删除", "sec_app_num_rev_accts":"删除", "sec_app_chargeoff_within_12_mths":"删除",
                  "sec_app_collections_12_mths_ex_med":"删除", "sec_app_mths_since_last_major_derog":"删除", "hardship_flag":"删除", "hardship_type":"删除", "hardship_reason":"删除",
                  "hardship_status":"删除", "deferral_term":"删除", "hardship_amount":"删除", "hardship_start_date":"删除", "hardship_end_date":"删除", "payment_plan_start_date":"删除",
                  "hardship_length":"删除", "hardship_dpd":"删除", "hardship_loan_status":"删除", "orig_projected_additional_accrued_interest":"删除", "hardship_payoff_balance_amount":"删除",
                  "hardship_last_payment_amount":"删除", "disbursement_method":"删除", "debt_settlement_flag":"删除", "debt_settlement_flag_date":"删除", "settlement_status":"删除",
                  "settlement_date":"删除", "settlement_amount":"删除", "settlement_percentage":"删除", "settlement_term":"删除"
                  }

for file in files:
    file_path = os.path.join(rawpath, file)
    LoadData = pd.read_excel(file_path)

    # 删除/保留列
    for column, choice in columns_judege.items():
        if choice == "删除":
            LoadData.drop(column, axis=1, inplace=True)
    
    # 删除包含缺失值的行
    LoadData.dropna(axis=0, how="any", inplace=True)

    selected_path = os.path.join("Data/ProcessingData", f"process_{file}")
    LoadData.to_excel(selected_path, index=False, header=True)

######################################## 特征编码 ########################################
processing_path = "Data/ProcessingData"
process_files = [f for f in os.listdir(processing_path) if f.endswith('.xlsx')]

# term 类别转换
term_mapping = {
    '36 months': '36',
    '60 months': '60'
}
# emp_length 类别转换
emp_length_mapping = {
    '< 1 year': 0,
    '1 year': 1,
    '2 years': 2,
    '3 years': 3,
    '4 years': 4,
    '5 years': 5,
    '6 years': 6,
    '7 years': 7,
    '8 years': 8,
    '9 years': 9,
    '10+ years': 10
}
# home_ownership 类别转换
home_ownership_mapping = {
    'ANY': 0,
    'MORTGAGE': 1,
    'RENT': 2,
    'OWN': 3
}
# loan_status 类别转换
loan_status_mapping = {
    'Current': 0,
    'Fully Paid': 1,
    'In Grace Period': 2,
    'Late (31-120 days)': 3,
    'Late (16-30 days)': 4
}

# 创建一个LabelEncoder对象
label_encoder = LabelEncoder()
# 创建一个空字典用于存储编码前后的对应关系
encoding_mapping = {}

for file in process_files:
    file_path = os.path.join(processing_path, file)
    processData = pd.read_excel(file_path)

    # 删除 loan_status 列中含有 Charged Off、Default、Issued的行
    processData = processData[~processData['loan_status'].isin(['Charged Off', 'Default', 'Issued'])]
    # 删除 purpose 列中含有 wedding 的行
    processData = processData[~processData['purpose'].str.contains('wedding')]

    processData['term'] = processData['term'].map(term_mapping)
    processData['emp_length'] = processData['emp_length'].map(emp_length_mapping)
    processData['home_ownership'] = processData['home_ownership'].map(home_ownership_mapping)
    processData['loan_status'] = processData['loan_status'].map(loan_status_mapping)
    for column in ['purpose', 'title', 'addr_state']:
        processData[column] = label_encoder.fit_transform(processData[column])
        encoding_mapping[column] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    processed_path = os.path.join("Data/ProcessedData", f"Final_{file}")
    processData.to_excel(processed_path, index=False, header=True)

# 输出 label_encoder 编码前后的对应关系
for file, mapping in encoding_mapping.items():
    print(f"Encoding Mapping for {file}:")
    for category, code in mapping.items():
        print(f"{category}: {code}")
    print('\n')

######################################## 数据标准化 ########################################
scale_path  = "Data/ProcessedData"
scale_files = [f for f in os.listdir(scale_path) if f.endswith(".xlsx")]
scaler = StandardScaler()

for file in scale_files:
    file_path = os.path.join(scale_path, file)
    scaleData = pd.read_excel(file_path)
    
    # 标准化特征列
    features = [col for col in scaleData.columns if col != 'loan_status']
    scaleData[features] = scaler.fit_transform(scaleData[features])

    # 保存目标列loan_status
    loan_status_column = scaleData['loan_status']
    scaleData = scaleData.drop(columns=['loan_status'], axis=1)
    scaleData['loan_status'] = loan_status_column

    scaled_file_path = os.path.join("Data/ScaledData", f"Scaled_{file}")
    scaleData.to_excel(scaled_file_path, index=False, header=True)

########### 至此，我们已经得到了三份纯数值文件，接下来为模型的训练。###########