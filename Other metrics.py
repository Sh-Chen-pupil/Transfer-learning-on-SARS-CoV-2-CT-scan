Yscore = model.predict(validation_generator)[:,1]
Yscore3 = model3.predict(validation_generator)[:,1]     
Ytrue=validation_generator.classes 
fpr, tpr, threshold = roc_curve(Ytrue, Yscore)           
roc_auc = auc(fpr, tpr)                                   
fpr3, tpr3, threshold3 = roc_curve(Ytrue, Yscore3)          
roc_auc3 = auc(fpr3, tpr3)   

optimal_threshold = threshold[np.argmax(tpr-fpr)]         
print('model best-threshold：',round(optimal_threshold,4))           

optimal_threshold3 = threshold[np.argmax(tpr3-fpr3)]        
print('model3 best-threshold：',round(optimal_threshold3,4)) 


Yhat=1.0*(Yscore>optimal_threshold)                       
yhat0=1.0*(Yscore<=optimal_threshold) 
TP=np.sum(Yhat*Ytrue)                     
FP=np.sum(Yhat*(1-Ytrue))              
TN=np.sum(yhat0*(1-Ytrue))                       
FN=np.sum(yhat0*Ytrue)


Yhat3=1.0*(Yscore3>optimal_threshold3)                      
yhat03=1.0*(Yscore3<=optimal_threshold3) 
TP3=np.sum(Yhat3*Ytrue)                    
FP3=np.sum(Yhat3*(1-Ytrue))               
TN3=np.sum(yhat03*(1-Ytrue))                       
FN3=np.sum(yhat03*Ytrue)


TPR = TP/(TP+FN)
print('MobileNetV2performances:\n')
print("Sensitivity :",round(TPR,4))
TNR = TN/(TN+FP)
print("Specificity :",round(TNR,4))
PPV = TP/(TP+FP)
NPV = TN/(TN+FN)
FPR = FP/(FP+TN)
FNR = FN/(TP+FN)
print("PPV :",round(PPV,4))
print("NPV :",round(NPV,4))
FDR = FP/(TP+FP)
ACC = (TP+TN)/(TP+FP+FN+TN)
print("Accuracy :",round(ACC,4))


TPR3 = TP3/(TP3+FN3)
print('VGG19 performances:\n')
print("Sensitivity :",round(TPR3,4))
TNR3 = TN3/(TN3+FP3)
print("Specificity :",round(TNR3,4))
PPV3 = TP3/(TP3+FP3)
NPV3 = TN3/(TN3+FN3)
FPR3 = FP3/(FP3+TN3)
FNR3 = FN3/(TP3+FN3)
print("PPV3 :",round(PPV3,4))
print("NPV3 :",round(NPV3,4))
FDR3 = FP3/(TP3+FP3)
ACC3 = (TP3+TN3)/(TP3+FP3+FN3+TN3)
print("Accuracy :",round(ACC3,4))
