import numpy as np
import pandas as pd
import locale

locale.setlocale(locale.LC_ALL, "")


def table_forecast(X_train: list, FORE: int, TRAIN: int):
    df_forecast = pd.DataFrame(columns=['Препарат', 'Точность прогноза'])
    for i in range(FORE):
        df_forecast.insert(2 + i, f'{X_train[TRAIN + i]}', 0)
    return df_forecast


def avg_3_last_years(y_train: list):
    s = 0
    kol = 0
    for i in y_train[-12:]:
        if i > 0.0001:
            s += i
            kol += 1
    if kol != 0:
        avg_y_train = s / kol
    else:
        avg_y_train = 0
    return avg_y_train


def x_new(X_train: list, FORE: int):
    for i in range(1, FORE + 1):
        if str(X_train[-1])[-1] == '4':
            X_train.append(str(int(str(X_train[-1])[-7:-2]) + 1) + '-' + '1')
        else:
            X_train.append(str(X_train[-1])[-7:-2] + '-' + str(int(str(X_train[-1])[-1]) + 1))
    return X_train


def if_recommended(y_train: list):
    if (abs((np.mean(y_train[-8:-4]) - np.mean(y_train[-4:])) / (
            np.mean(y_train[-8:-4]) + np.mean(y_train[-4:]))) < 0.17) and (
            abs((np.mean(y_train[-12:-8]) - np.mean(y_train[-8:-4])) / (
                    np.mean(y_train[-12:-8]) + np.mean(y_train[-8:-4]))) < 0.2) and abs(
        (np.mean(y_train[-12:-4]) - np.mean(y_train[-4:])) / (
                np.mean(y_train[-12:-4]) + np.mean(y_train[-4:]))) < 0.15 and abs(
        (np.mean(y_train[-1:]) - np.mean(y_train[-2:-1])) / (
                np.mean(y_train[-1:]) + np.mean(y_train[-2:-1]))) < 0.3:
        is_rec = 1
        accur = '<=35%'
        comment = str('прогноз рекомендуется(sum_diff<35%, mape<35%)')
    else:
        is_rec = 0
        accur = '>35%'
        comment = str('прогноз не рекомендуется(sum_diff может быть >35%, mape может быть >35%)')
    return is_rec, accur, comment
