import pandas as pd
import numpy as np

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

HDData = pd.read_csv("Dataset7.csv")
HDData = HDData.replace("?",np.nan)

model = BayesianModel([('age','fbs'),('age','trestbps'),('sex','trestbps'),('exang','trestbps'),('fbs','HeartDisease'),('trestbps','HeartDisease'),('HeartDisease','chol'),('HeartDisease','thalach'),('HeartDisease','restecg')])

model.fit(HDData,estimator=MaximumLikelihoodEstimator)

HD_Infer = VariableElimination(model)
q = HD_Infer.query(variables=['HeartDisease'],evidence={'age':40})
print("\n")
print("The Probablity of Heart Disease give age=40:")
print(q['HeartDisease'])
print()

