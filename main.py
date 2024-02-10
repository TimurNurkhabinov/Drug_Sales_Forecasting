import streamlit as st
import numpy as np
import pandas as pd
import warnings
from ml.models.exponential_smoothing.model import Forecast
from ml.data_prepare import data_preprocessing
from scripts.utils import to_excel, checkbox_container
import locale

st.set_page_config(layout="wide",
                   page_title="Прогноз продаж препаратов"
                   )


locale.setlocale(locale.LC_ALL, "")

warnings.filterwarnings('ignore')

st.write(""" 
         # Прогнозирование продаж препаратов
""")

st.sidebar.header('Пользовательские данные')

uploaded_file = st.sidebar.file_uploader("Прикрепите ваш excel-файл с данными", type='xlsx')

if uploaded_file:
    result = None
    df_old = pd.read_excel(uploaded_file)
    df_old = df_old.set_index('Unnamed: 0')
    df_old = df_old.transpose()
    df_old.reset_index(inplace=True)
    df_old = df_old.rename(columns={'index': 'Дата'})
    df_old['Дата'] = pd.to_datetime(df_old['Дата'], format='%Y-%m')

    st.subheader('Пользовательские данные')
    st.write(df_old)

    def get_selected_checkboxes():
        return [i.replace('dynamic_checkbox_', '') for i in st.session_state.keys() if
            i.startswith('dynamic_checkbox_') and st.session_state[i]]

    with st.form("forecast_form"):
        st.header("Заполните данные для прогноза")
        FORECAST_PERIOD = st.selectbox('Выберите на сколько кварталов вперед сделать прогноз', (1, 2, 3, 4))

        checkbox_container(df_old.columns[1:])
        new_data = get_selected_checkboxes()
        new_data.insert(0, "Дата")
        sel_df=data_preprocessing(df_old,list(df_old.columns)[1:])
        sel_df=sel_df.T
        sel_df.reset_index(inplace = True)
        sel_df=sel_df.T
        sel_df=sel_df.reset_index(drop = True)
        sel_df=sel_df.set_index(0)
        sel_df=sel_df.T
        sel_df=sel_df.rename(columns={'Y-quart':'Препарат'})
        sel_df=sel_df[:-2]
        #sel_df.head(5)
        make_predict_button = st.form_submit_button("Сделать прогноз")

        if make_predict_button:
            #st.write(sel_df)
            forecast = Forecast(df_old[new_data], FORECAST_PERIOD, 1, len(new_data) - 1)
            result = forecast.predict()
            result=sel_df.merge(result, on='Препарат', how='left')
            result = result.loc[((result['Точность прогноза'] ==">35%")| (result['Точность прогноза'] =="<=35%"))]
            result=result.reset_index()
            result=result.drop(columns='index')

    if result is not None:
        st.subheader('Прогноз')
        st.write(result)
        df_xlsx = to_excel(result)
        st.download_button(label='📥 Скачать файл в формате excel',
                           data=df_xlsx,
                           file_name='Prediction.xlsx')
