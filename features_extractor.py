import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, history_path: str, users_path: str):
        """
        Конструктор класса. Считывает данные из файлов history.tsv и users.tsv.
        """
        self.history = pd.read_csv(history_path, sep='\t')
        self.users = pd.read_csv(users_path, sep='\t')

    def _get_history_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Возвращает информацию из истории для каждой строки датафрейма.
        Вычисляет avg_time_between_ads, history_cpm_mean и adv_shown_freq.
        """
        history_features = []
        for index, row in df.iterrows():
            users_ids = [int(i) for i in row['user_ids'].split(',')]
            users_history_shown = self.history[
                (self.history['user_id'].isin(users_ids)) & 
                (self.history['hour'] < row['hour_start'])
            ]
            # Средняя цена показа
            cpm_mean = users_history_shown['cpm'].mean()
            # Частота показов
            adv_shown_freq = users_history_shown['publisher'].count() / len(users_ids) if len(users_ids) > 0 else 0
            # Среднее время между показами
            avg_time_between_ads = 0.0
            for user_id, group in users_history_shown.groupby('user_id'):
                hours = sorted(group['hour'].unique())
                if len(hours) > 1:
                    avg_time_between_ads += np.mean(np.diff(hours))
            avg_time_between_ads /= len(users_ids) if len(users_ids) > 0 else 1
            # Добавляем фичи
            history_features.append({
                'history_cpm_mean': cpm_mean if not np.isnan(cpm_mean) else 0,
                'adv_shown_freq': adv_shown_freq,
                'avg_time_between_ads': avg_time_between_ads
            })
        return pd.DataFrame(history_features, index=df.index)

    def _get_session_info(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает среднюю продолжительность сессии (avg_session_duration).
        """
        avg_session_durations = []
        for index, row in df.iterrows():
            users_ids = [int(i) for i in row['user_ids'].split(',')]
            users_history_shown = self.history[
                (self.history['user_id'].isin(users_ids)) & 
                (self.history['hour'] < row['hour_start'])
            ]
            total_duration = 0
            session_count = 0
            for user_id, group in users_history_shown.groupby('user_id'):
                group = group.sort_values('hour')
                sessions = []
                current_session = [group['hour'].iloc[0]]
                for hour in group['hour'].iloc[1:]:
                    if hour - current_session[-1] > 6:
                        sessions.append(current_session)
                        current_session = [hour]
                    else:
                        current_session.append(hour)
                sessions.append(current_session)
                session_count += len(sessions)
                total_duration += sum(max(session) - min(session) for session in sessions)
            avg_session_duration = total_duration / session_count if session_count > 0 else 0
            avg_session_durations.append(avg_session_duration)
        return pd.Series(avg_session_durations, index=df.index)

    def get_p1(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает вероятность показа на площадке 1 (p1).
        """
        publishers_amount = 21
        p1 = []
        for index, row in df.iterrows():
            publishers = [int(i) for i in row['publishers'].split(',')]
            p1.append(1 if 1 in publishers else 0)
        return pd.Series(p1, index=df.index)

    def get_p2(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает вероятность показа на площадке 2 (p2).
        """
        publishers_amount = 21
        p2 = []
        for index, row in df.iterrows():
            publishers = [int(i) for i in row['publishers'].split(',')]
            p2.append(1 if 2 in publishers else 0)
        return pd.Series(p2, index=df.index)

    def get_delay(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает разницу между hour_end и hour_start.
        """
        if 'delay' not in df.columns:
            df['delay'] = df['hour_end'] - df['hour_start']
        return df['delay']

    def get_remaining_time_to_next_ad(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает остаточное время до следующего показа.
        """
        delay = self.get_delay(df)
        history_info = self._get_history_info(df)
        remaining_time = delay - history_info['avg_time_between_ads']
        return remaining_time.clip(lower=0)

    def get_cpm(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает цену за показ (cpm).
        """
        if 'cpm' not in df.columns:
            raise KeyError("Колонка 'cpm' отсутствует в датафрейме.")
        return df['cpm']

    def get_avg_session_duration(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает среднюю продолжительность сессии.
        """
        if 'avg_session_duration' not in df.columns:
            df['avg_session_duration'] = self._get_session_info(df)
        return df['avg_session_duration']

    def get_history_cpm_mean(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает среднюю цену показа в истории.
        """
        history_info = self._get_history_info(df)
        if 'history_cpm_mean' not in df.columns:
            df['history_cpm_mean'] = history_info['history_cpm_mean']
        return df['history_cpm_mean']

    def get_adv_shown_freq(self, df: pd.DataFrame) -> pd.Series:
        """
        Возвращает частоту показа рекламы в истории.
        """
        history_info = self._get_history_info(df)
        if 'adv_shown_freq' not in df.columns:
            df['adv_shown_freq'] = history_info['adv_shown_freq']
        return df['adv_shown_freq']

    def get_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Собирает все фичи в один датафрейм.
        """
        features = {
            'p1': self.get_p1(df),
            'remaining_time_to_next_ad': self.get_remaining_time_to_next_ad(df),
            'delay': self.get_delay(df),
            'cpm': self.get_cpm(df),
            'p2': self.get_p2(df),
            'avg_session_duration': self.get_avg_session_duration(df),
            'history_cpm_mean': self.get_history_cpm_mean(df),
            'adv_shown_freq': self.get_adv_shown_freq(df)
        }
        return pd.DataFrame(features)