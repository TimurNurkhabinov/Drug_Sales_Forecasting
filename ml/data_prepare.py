import pandas as pd
import locale
import numpy as np

locale.setlocale(locale.LC_ALL, "")


def data_preprocessing(df: pd.DataFrame, drugs: list):
    df = df.fillna(0.00001)
    for i in drugs:
        df.loc[df[f'{i}'] <= 0, i] = 0.00001
    dfq = df

    dfq['month'] = dfq['Дата'].apply(lambda x: int(x.month))
    dfq['Num_mon_in_quart']=dfq['month'].apply(lambda x: x%3)
    if np.array(dfq['Num_mon_in_quart'])[-1]==2:
        dfq = dfq.iloc [:-1 , :]
    if np.array(dfq['Num_mon_in_quart'])[-1]==1:
        dfq = dfq.iloc [:-1 , :]

    dfq['quarter'] = dfq['Дата'].apply(lambda x: str(x.quarter))
    dfq['year'] = dfq['Дата'].apply(lambda x: str(x.year))
    dfq['Y-quart'] = dfq[['year', 'quarter']].agg('-'.join, axis=1)
    dfq = dfq.groupby(['Y-quart']).sum()
    dfq.reset_index(inplace=True)
    return dfq
