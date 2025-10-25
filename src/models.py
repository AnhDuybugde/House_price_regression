from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

def get_models():
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
