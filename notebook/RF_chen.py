import numpy as np
from sklearn.ensemble import RandomForestRegressor

size = 10000
np.random.seed(10)
X_seed = np.random.normal(0,1,size)
X0 = X_seed + np.random.normal(0, .1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X = np.array([X0, X1, X2]).T
Y = X0 + X1 + X2

rf = RandomForestRegressor(n_estimators=20, max_features=2)
rf.fit(X, Y)
l = map(lambda x:round(x, 3), rf.feature_importances_)
print("Scores for X0, X1, X2:", list(l))

3>=2
