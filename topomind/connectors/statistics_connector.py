import numpy as np
from typing import Dict, Any
from topomind.connectors.base import ExecutionConnector

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


class StatisticsConnector(ExecutionConnector):

    def execute(self, tool, args: Dict[str, Any], timeout: int)-> Any:

        op = args["operation"].strip().upper()

        # ================= DESCRIPTIVE =================

        if op == "MEAN":
            return {"result": float(np.mean(args["values"]))}

        elif op == "STD_DEV_SAMPLE":
            return {"result": float(np.std(args["values"], ddof=1))}

        elif op == "STD_DEV_POPULATION":
            return {"result": float(np.std(args["values"], ddof=0))}

        elif op == "VARIANCE":
            return {"result": float(np.var(args["values"]))}

        elif op == "MEDIAN":
            return {"result": float(np.median(args["values"]))}

        # ================= NORMALIZATION =================

        elif op == "Z_SCORE":
            values = np.array(args["values"])
            return {"result": list(stats.zscore(values))}

        elif op == "COEFFICIENT_OF_VARIATION":
            values = np.array(args["values"])
            return {"result": float(np.std(values) / np.mean(values))}

        # ================= RELATIONSHIP =================

        elif op == "COVARIANCE":
            x = args["x"]
            y = args["y"]
            return {"result": float(np.cov(x, y)[0][1])}

        elif op == "CORRELATION":
            x = args["x"]
            y = args["y"]
            return {"result": float(np.corrcoef(x, y)[0, 1])}

        # ================= REGRESSION =================

        elif op in {
            "TREND_SLOPE",
            "REGRESSION_INTERCEPT",
            "R_SQUARED",
            "ADJUSTED_R_SQUARED",
            "TIME_SERIES_R_SQUARED",
            "REGRESSION_MODEL",
        }:

            x = np.array(args["x"]).reshape(-1, 1)
            y = np.array(args["y"])

            model = LinearRegression().fit(x, y)

            if op == "TREND_SLOPE":
                return {"result": float(model.coef_[0])}

            elif op == "REGRESSION_INTERCEPT":
                return {"result": float(model.intercept_)}

            elif op == "R_SQUARED":
                return {"result": float(model.score(x, y))}

            elif op == "ADJUSTED_R_SQUARED":
                r2 = model.score(x, y)
                n = len(y)
                p = 1
                adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                return {"result": float(adj)}

            elif op == "TIME_SERIES_R_SQUARED":
                return {"result": float(model.score(x, y))}

            elif op == "REGRESSION_MODEL":
                return {
                    "result": {
                        "slope": float(model.coef_[0]),
                        "intercept": float(model.intercept_),
                    }
                }

        # ================= TIME SERIES DIAGNOSTICS =================

        elif op == "AUTOCORRELATION":
            values = np.array(args["values"])
            lag = args.get("lag", 1)
            return {
                "result": float(
                    np.corrcoef(values[:-lag], values[lag:])[0, 1]
                )
            }

        elif op == "AUTOCORRELATION_PROBABILITY":
            values = np.array(args["values"])
            lag = args.get("lag", 1)
            r = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            n = len(values)
            t = r * np.sqrt((n - 2) / (1 - r**2))
            p = 2 * (1 - stats.t.cdf(abs(t), df=n - 2))
            return {"result": float(p)}

        elif op == "LJUNG_BOX":
            values = np.array(args["values"])
            lag = args.get("lag", 1)
            lb = acorr_ljungbox(values, lags=[lag], return_df=True)
            return {"result": float(lb["lb_stat"].values[0])}

        # ================= ERROR METRICS =================

        elif op == "RMSE":
            actual = np.array(args["actual"])
            predicted = np.array(args["predicted"])
            return {"result": float(np.sqrt(np.mean((actual - predicted) ** 2)))}

        elif op == "MAPE":
            actual = np.array(args["actual"])
            predicted = np.array(args["predicted"])
            return {
                "result": float(
                    np.mean(np.abs((actual - predicted) / actual)) * 100
                )
            }

        # ================= DATA PREPROCESSING =================

        elif op == "CLEAN_START_INDEX":
            values = args["values"]
            for i, v in enumerate(values):
                if v is not None:
                    return {"result": i}
            return {"result": 0}

        # ================= ANOMALY =================

        elif op == "OUTLIER_DETECTION":
            values = np.array(args["values"])
            z = np.abs(stats.zscore(values))
            threshold = args.get("threshold", 3)
            return {"result": list(np.where(z > threshold)[0])}

        else:
            raise ValueError(f"Unsupported StatOperation: {op}")
