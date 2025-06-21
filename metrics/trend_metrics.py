import pandas as pd

class TrendMetrics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._validate_columns()

        self.delta_true = self.df['Freight_Price'].diff().iloc[1:]
        self.delta_pred = self.df['yhat_exp'].diff().iloc[1:]

    def _validate_columns(self):
        required = {'Freight_Price', 'yhat_exp'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"В DataFrame отсутствуют необходимые колонки: {missing}")

    def tan_error(self) -> tuple:
        diffs = (self.delta_pred - self.delta_true).abs()
        return diffs.mean(), diffs.min(), diffs.max()

    def directional_accuracy(self) -> float:
        correct = (self.delta_pred * self.delta_true >= 0).sum()
        total = len(self.delta_true)
        return correct / total if total else 0.0

    def summary(self) -> dict:
        mean_diff, min_diff, max_diff = self.tan_error()
        return {
            'Mean Tangent Error': mean_diff,
            'Min Tangent Error': min_diff,
            'Max Tangent Error': max_diff,
            'Directional Accuracy': self.directional_accuracy()
        }