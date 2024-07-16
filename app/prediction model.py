import os
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap
import streamlit as st
import numpy as np
from streamlit_shap import st_shap

# 定义输出路径
output_path = "F:/python_project/output"
os.makedirs(output_path, exist_ok=True)

# 读取数据
train_data = pd.read_excel(r"D:\文档\Dr.wu\SAH基于NIS\I60主整合（整理数据）\6.模型确定\1.建模数据\7train.xlsx")
validation_data = pd.read_excel(r"D:\文档\Dr.wu\SAH基于NIS\I60主整合（整理数据）\6.模型确定\1.建模数据\7validation.xlsx")

# 定义分类变量和连续变量
binary_vars = ["Died_during_hospitalization", "Hydrocephalus", "Cerebral_vasospasm_and_vasoconstriction",
               "Mechanical_ventilation_greater_than_96_consecutive_hours"]
multi_class_vars = ["Hospital_discharge_transfer_indicator"]
continuous_vars = ["Age", "Total_number_of_diagnoses"]

# 数据预处理
train_data[binary_vars] = train_data[binary_vars].astype('int')
validation_data[binary_vars] = validation_data[binary_vars].astype('int')
train_data[multi_class_vars] = train_data[multi_class_vars].astype('category')
validation_data[multi_class_vars] = validation_data[multi_class_vars].astype('category')

X_train = train_data.drop(columns=['group'])
y_train = train_data['group']
X_valid = validation_data.drop(columns=['group'])
y_valid = validation_data['group']

# CatBoost 模型
cat_features = [X_train.columns.get_loc(var) for var in multi_class_vars]
model = CatBoostClassifier(depth=5, iterations=300, learning_rate=0.05, l2_leaf_reg=6, border_count=96, random_state=42)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
model.fit(train_pool, eval_set=valid_pool, verbose=100)

# 生成 SHAP 值
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(valid_pool)
base_value = explainer.expected_value
feature_names = X_valid.columns

# 使用简称
short_feature_names = {
    'Age': 'Age',
    'Hospital_discharge_transfer_indicator': 'Transfer',
    'Died_during_hospitalization': 'Died',
    'Hydrocephalus': 'Hydrocephalus',
    'Cerebral_vasospasm_and_vasoconstriction': 'Vasospasm',
    'Total_number_of_diagnoses': 'Diagnoses',
    'Mechanical_ventilation_greater_than_96_consecutive_hours': 'MechVent'
}

# 初始化SHAP的JS库
shap.initjs()

# 创建 Streamlit 应用
st.title('Final prediction model for extended LOS in patients with SAH')
st.write(
    "The patient information is entered to predict the probability of his/her LOS being prolonged and a SHAP plot is generated to interpret the prediction.")

# 定义映射
transfer = {0: 'Not transferred', 1: 'To acute care hospital', 2: 'To other facility'}
yes_no = {0: 'No', 1: 'Yes'}

# 输入表单
with st.form("patient_info_form"):
    age = st.number_input("Age", value=88, step=1)
    transfer = st.selectbox("Hospital discharge transfer indicator", options=list(transfer.keys()), format_func=lambda x: transfer[x])
    died = st.selectbox("Died during hospitalization", options=list(yes_no.keys()), format_func=lambda x: yes_no[x])
    hydrocephalus = st.selectbox("Hydrocephalus", options=list(yes_no.keys()), format_func=lambda x: yes_no[x])
    vasospasm = st.selectbox("Cerebral vasospasm and vasoconstriction", options=list(yes_no.keys()), format_func=lambda x: yes_no[x])
    diagnoses = st.number_input("Total number of diagnoses", value=1, step=1)
    mech_vent = st.selectbox("Mechanical ventilation greater than 96 consecutive hours", options=list(yes_no.keys()), format_func=lambda x: yes_no[x])

    submitted = st.form_submit_button("Predict", use_container_width=True)
    if submitted:
        # 构建输入数据
        input_data = {
            'Age': age,
            'Hospital_discharge_transfer_indicator': transfer,
            'Died_during_hospitalization': died,
            'Hydrocephalus': hydrocephalus,
            'Cerebral_vasospasm_and_vasoconstriction': vasospasm,
            'Total_number_of_diagnoses': diagnoses,
            'Mechanical_ventilation_greater_than_96_consecutive_hours': mech_vent
        }
        input_df = pd.DataFrame([input_data])

        # 预测概率
        prediction = model.predict_proba(input_df)[:, 1][0]
        st.markdown(f"<h5>Based on feature values, predicted possibility of extended LOS is {prediction:.2%}</h5>", unsafe_allow_html=True)

        # 生成SHAP值
        shap_values_single = explainer.shap_values(input_df)
        st_shap(shap.force_plot(base_value, shap_values_single[0], input_df, feature_names=[short_feature_names[col] for col in input_df.columns]))
