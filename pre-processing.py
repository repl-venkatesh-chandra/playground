# -*- coding: utf-8 -*-
"""
@author: vchan
"""

import pandas
from sklearn.preprocessing import StandardScaler

risk_df = pandas.read_csv("data_raw.csv")

#Subset the required columns
X = risk_df.iloc[:,2:5]
y = risk_df['risk']

#Standardize the columns

standardizer = StandardScaler()
X_std = standardizer.fit_transform(X)

#Save output
X_std.to_csv("X.csv")
y.to_csv("y.csv")
