from statsmodels.tsa.api import ExponentialSmoothing
import numpy as np
import pandas as pd
from ml.data_prepare import data_preprocessing
from ml.models.exponential_smoothing.utils import x_new, avg_3_last_years, table_forecast, if_recommended
import locale

locale.setlocale(locale.LC_ALL, "")


class Forecast:
    def __init__(self, data: pd.DataFrame, FORECAST_PERIOD: int, MIN_ID: int, MAX_ID: int):
        self.dataframe = data
        self.FORECAST_PERIOD = FORECAST_PERIOD
        self.MIN_ID = MIN_ID
        self.MAX_ID = MAX_ID

    def predict(self) -> pd.DataFrame:
        FORE = self.FORECAST_PERIOD
        MIN_ID = self.MIN_ID
        MAX_ID = self.MAX_ID
        drugs = list(self.dataframe.columns)[MIN_ID:MAX_ID + 1]  # Список препаратов
        dfq = data_preprocessing(self.dataframe,
                                 drugs)  # Заменяем пустые и отрицательные значения, группируем по кварталам
        TRAIN = dfq.shape[0]
        X_train = x_new(list(dfq['Y-quart'][0:TRAIN]), FORE)  # Добавляем в список кварталов прогнозируемые кварталы
        df_forecast = table_forecast(X_train, FORE, TRAIN)  # Создаем финальную таблицу с прогнозами
        # параметры модели:
        season_type = 'multiplicative'
        seas_period_ = 4
        params = [0.00001, 0.00002, 0.00005, 0.00007, 0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.002, 0.003, 0.004, 0.005,
                  0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3,
                  1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]
        j = 0
        for drug in drugs:
            dfq_d = dfq[['Y-quart', drug]]
            y_train = list(dfq[drug][0:TRAIN])
            avg_y_train = avg_3_last_years(y_train)  # средняя координата ненулевых кварталов за последние 3 года
            best_param = 0
            min_dist = 10000000
            alp = 0
            # Определение параметра модели alpha(smoothing_level):
            for smoot_level in params:
                fit1 = ExponentialSmoothing(y_train, seasonal_periods=seas_period_, trend='additive',
                                            seasonal=season_type, damped=True).fit(smoothing_level=smoot_level)
                fitted_ = fit1.predict(0, len(y_train) + FORE - 1)
                avg_y_pred = np.mean(fitted_[TRAIN:TRAIN + FORE])
                if abs(avg_y_train - avg_y_pred) < min_dist:
                    alp = smoot_level
                    min_dist = abs(avg_y_train - avg_y_pred)

            is_rec, accur, comment = if_recommended(y_train)  # Определение, рекомендуется ли к прогнозированию
            # Обучаем модель:
            fit1 = ExponentialSmoothing(y_train, seasonal_periods=seas_period_, trend='additive', seasonal=season_type,
                                        damped=True).fit(smoothing_level=alp)
            fitted_ = fit1.predict(0, len(y_train) + FORE - 1)
            # Если полученный прогноз меньше 0, то присваиваем 0:
            for i in range(FORE):
                if float(fitted_[TRAIN + i]) < 0:
                    fitted_[TRAIN + i] = 0
            # Если последние 3 квартала исходных данных нулевые, то прогноз делаем 0:
            if sum(y_train[-3:]) < 0.0001:
                for i in range(FORE):
                    fitted_[len(fitted_) - i - 1] = 0

            # Записываем прогноз препарата в таблицу:
            line = [drug, accur]
            for i in range(FORE):
                line.append(fitted_[TRAIN + i])
            df_forecast.loc[j] = line
            j += 1
        return df_forecast
