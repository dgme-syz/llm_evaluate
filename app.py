# app.py
import streamlit as st
import json
import pandas as pd

st.set_page_config(page_title="翻译 JSONL 可视化", layout="wide")
st.title("翻译 JSONL 可视化")

# 上传文件
uploaded_file = st.file_uploader("上传 JSONL 文件", type=["jsonl"])

if uploaded_file is not None:
    # 解析 JSONL
    data_list = []
    for line in uploaded_file:
        item = json.loads(line)
        # 处理每条记录，取需要显示的字段
        data_list.append({
            "源语言": item.get("extra_info", {}).get("src_lang", ""),
            "目标语言": item.get("extra_info", {}).get("tgt_lang", ""),
            "原文": item.get("extra_info", {}).get("src", ""),
            "模型输出": "\n".join(item.get("response", [])),
            "COMET-22": item.get("wmt24_comet-22_score_per_example", [""])[0],
            "COMETKiwi": item.get("wmt24_cometkiwi_score_per_example", [""])[0],
            "BLEURT-22": item.get("wmt24_BLEURT_score_per_example", [""])[0]
        })
    
    df = pd.DataFrame(data_list)
    
    st.subheader("原始数据表")
    # 搜索和过滤
    search_text = st.text_input("搜索原文或输出中的关键词")
    filtered_df = df[df["原文"].str.contains(search_text, case=False, na=False) |
                     df["模型输出"].str.contains(search_text, case=False, na=False)] if search_text else df
    
    st.dataframe(filtered_df, use_container_width=True)
    
    st.subheader("指标可视化")
    metrics_to_plot = ["COMET-22", "COMETKiwi", "BLEURT-22"]
    st.bar_chart(filtered_df[metrics_to_plot])