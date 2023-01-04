from dataprocessing import genexpr_txt_file_to_csv,genexpr_anotation_txt_to_csv
import pandas as pd


target1= genexpr_anotation_txt_to_csv("DataGenExpression/dataset_A1_annotation.txt","DataGenExpression/Annotation_Dataset_1.csv")
target2= genexpr_anotation_txt_to_csv("DataGenExpression/dataset_A2_annotation.txt","DataGenExpression/Annotation_Dataset_2.csv")
target3= genexpr_anotation_txt_to_csv("DataGenExpression/dataset_A3_annotation.txt","DataGenExpression/Annotation_Dataset_3.csv")
data1= genexpr_txt_file_to_csv("DataGenExpression/GSE122505_Dataset_1_matrix.txt","DataGenExpression/GSE122515_Dataset_1.csv")
data2= genexpr_txt_file_to_csv("DataGenExpression/GSE122511_Dataset_2_matrix.txt","DataGenExpression/GSE122515_Dataset_2.csv")
data3= genexpr_txt_file_to_csv("DataGenExpression/GSE122515_Dataset_3_ensembl.txt","DataGenExpression/GSE122515_Dataset_3.csv")



dt1 = data1.join(target1)
dt2= data2.join(target2)
dt3 = data3.join(target3)

#all_samples = pd.concat([dt1,dt3],join="inner")
dt1.to_csv("DataGenExpression/Dataset1.csv")
dt2.to_csv("DataGenExpression/Dataset2.csv")
dt3.to_csv("DataGenExpression/Dataset3.csv")
#all_samples.to_csv("DataGenExpression/Alldata.csv")

