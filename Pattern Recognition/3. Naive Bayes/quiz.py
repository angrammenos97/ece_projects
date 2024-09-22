from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy

data = pd.read_csv("quiz_data.csv")

fprM1, tprM1, thresholdsM1 = roc_curve(data.Class, data.P_M1)
print(thresholdsM1)
print(tprM1)
###############

##############
fprM2, tprM2, thresholds = roc_curve(data.Class, data.P_M2)
print("AUC: ", auc(fprM2, tprM2))
##############
plt.plot(fprM1, tprM1, 'b')
plt.plot(fprM2, tprM2, 'g')
plt.plot([0,1],[0,1],'r')
plt.show()