import csv
from tensorflow import keras
from numpy import loadtxt
import numpy as np

import pandas as pd
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
import inferDataPreconditionDemo

import time
time_startpre= time.time()

model = inferDataPreconditionDemo.model
d_test = inferDataPreconditionDemo.d_test

WPdict = inferDataPreconditionDemo.WPdictionary

#load dataset Mobile unseen test dataset
# d_test = './data/unseen.csv'

file = d_test
with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    fields =reader.fieldnames
    print(reader.fieldnames)
    features = fields[:20]
    features = fields
    print(features)
df = pd.read_csv(file, sep=',')
print(df.values)

X = df.iloc[:,:20].values
y = df.iloc[:, 20:-1].values
rslt_df = df

print(df)
print(rslt_df)

print(rslt_df.size)
#drop last column
# rslt_df = rslt_df.iloc[:,:-1]
# actual = df[df.columns[-1]]

print(rslt_df)
# For MP dataset. get actual from encoded values
# actual = df[df.columns[-1]]
actual = list()
for i in range(len(y)):
    actual.append(np.argmax(y[i]))

ActualOutcome = actual
rslt_df = df
print(rslt_df.size)
#drop last column
rslt_df = rslt_df.iloc[:,:20]
print(rslt_df)


for key, value in WPdict.items():
    print(key, '->', value)
    rslt_df[key] = rslt_df[key].map(lambda x: "Y" if x >= value else "N")
    print(rslt_df)

WP_df = rslt_df
print(WP_df)
WP_violation_count =WP_df.apply(lambda x: x.str.contains("N")).sum()

WP_satisfied_count =WP_df.apply(lambda x: x.str.contains("Y")).sum()

WP_df['total_Sat'] = 0

print(WP_df)
for key, value in WPdict.items():
    print(key, '->', value)
    WP_df['total_Sat'] = WP_df['total_Sat']+ WP_df[key].str.contains('Y', regex=False).astype(int)
    print(WP_df)


WP_df['total_Viol'] = 0
for key, value in WPdict.items():
    print(key, '->', value)
    WP_df['total_Viol'] = WP_df['total_Viol']+ WP_df[key].str.contains('N', regex=False).astype(int)
    print(WP_df)


violationMean= WP_df['total_Viol'].mean()

print(violationMean)

satisfiedMean = WP_df['total_Sat'].mean()

print(satisfiedMean)

#Check important features
print("WP_violation_count")
print(WP_violation_count)
print("WP_satisfied_count")
print(WP_satisfied_count)

#important feature detection
important_features = []
unimportant_featues = []
print(WP_violation_count.mean())
print(WP_satisfied_count.mean())

for i, v in WP_violation_count.items():
    #print('index: ', i, 'value: ', v)
    if (v <= WP_violation_count.mean()):
    #if (v <= violationMean):
        print('index: ', i, 'value: ', v)
        important_features.append(str(i))
    else:
        unimportant_featues.append(str(i))

print("More_Important_features: ")
print(important_features)
print("Less_Important_features: ")
print(unimportant_featues)

WP_df["vCount_MoreImpFeat"] = 0
WP_df["vCount_LessImpFeat"] = 0
for i in  important_features:
    WP_df["vCount_MoreImpFeat"] = WP_df["vCount_MoreImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)
for i in  unimportant_featues:
    WP_df["vCount_LessImpFeat"] = WP_df["vCount_LessImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)

print(WP_df)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Correct" if (x['vCount_MoreImpFeat'] == 0) else "Uncertain", axis=1)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Wrong" if (x['vCount_LessImpFeat'] > violationMean and x['vCount_MoreImpFeat'] < violationMean and x['vCount_LessImpFeat'] != x['vCount_MoreImpFeat']) else "Correct", axis=1)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Uncertain" if (x['vCount_LessImpFeat'] == x['vCount_MoreImpFeat'] and x['vCount_MoreImpFeat'] != 0) else x["DeepInfer_Implication"], axis=1)

#Appending ActualOutcome Column
print(ActualOutcome)
WP_df['Actual_Outcome'] = ActualOutcome
print(WP_df)

#For PIMA DIABETES
# Danny
predictionvalue = list()
y_p = model.predict(X)
for i in range(len(y_p)):
    predictionvalue.append(np.argmax(y_p[i]))
# predictionvalue = (model.predict(X) > 0.5).astype(int) ## Danny

#predictionvalue
WP_df['Predicted_Outcome'] = predictionvalue
print(WP_df)


WP_df["GroundTruth"] = WP_df.apply(lambda x: "Correct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "Wrong", axis=1)

Total_GT_Correct = WP_df["GroundTruth"].str.contains('Correct', regex=False).sum().astype(int)
Total_GT_Wrong = WP_df["GroundTruth"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Correct = WP_df["DeepInfer_Implication"].str.contains('Correct', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Wrong = WP_df["DeepInfer_Implication"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Uncertain = WP_df["DeepInfer_Implication"].str.contains('Uncertain', regex=False).sum().astype(int)

print("Total Violation:", WP_df['total_Viol'].sum())
print("Total Satisfied:", WP_df['total_Sat'].sum())
print("Total_GroundTruth_Correct:",Total_GT_Correct)
print("Total_GroundTruth_Incorrect:",Total_GT_Wrong)
print("Total_DeepInfer_Implication_Correct:",Total_DeepInfer_Implication_Correct)
print("Total_DeepInfer_Implication_Incorrect:",Total_DeepInfer_Implication_Wrong)
print("Total Uncertain:",Total_DeepInfer_Implication_Uncertain)

elapsed_timePerm = time.time() - time_startpre

print(elapsed_timePerm)





