from pyreadr import read_r
import pandas as pd
# Reads data from Rdata frame and saves single datasets as well as whole dataset
data = read_r("../DataGenExpression/Datasets.RData")
dataframes = []
for i in range(1,4):
    dataframes.append(data[f"data.{i}"].T.join(data[f"info.{i}"]))
    dataframes[i-1].to_csv(f"../DataGenExpression/Dataset{i}.csv")
all_data = pd.concat(dataframes)
all_data.to_csv("../DataGenExpression/Alldata.csv")

