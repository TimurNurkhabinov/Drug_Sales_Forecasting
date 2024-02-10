from io import BytesIO
import pandas as pd
import streamlit as st
import numpy as np


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'})
    worksheet.set_column('A:A', None, format1)
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def checkbox_container(drugs) -> None:
    st.subheader('Выберите препараты')
    select_all = st.checkbox("Выбрать все")
    if select_all:
        for i in drugs:
            st.session_state['dynamic_checkbox_' + i] = True
    for i in drugs:
        st.checkbox(i, key='dynamic_checkbox_' + i)

