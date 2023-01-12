from utils.rawdataprocessing_utils import genexpr_txt_file_to_csv,genexpr_anotation_txt_to_csv
from pyreadr import read_r
import pandas as pd
data = read_r("../DataGenExpression/Datasets.RData")
dataframes = []
for i in range(1,4):
    dataframes.append(data[f"data.{i}"].T.join(data[f"info.{i}"]))
    dataframes[i-1].to_csv(f"../DataGenExpression/Dataset{i}.csv")
all_data = pd.concat(dataframes)
all_data.to_csv("../DataGenExpression/Alldata.csv")

