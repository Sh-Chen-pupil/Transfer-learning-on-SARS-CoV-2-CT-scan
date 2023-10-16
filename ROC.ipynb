Yscore =model.predict(validation_generator)[:,1]   
Yscore2=model2.predict(validation_generator)[:,1]
Yscore3=model3.predict(validation_generator)[:,1]
Ytrue=validation_generator.classes         


def multi_models_roc(names, sampling_methods, colors,Ytrue, save=True, dpin=100):
    plt.figure(figsize=(5, 5), dpi=dpin)
    for (name, y_pred, colorname) in zip(names, sampling_methods, colors):
        
      
        fpr, tpr, thresholds = roc_curve(Ytrue, y_pred, pos_label=1)
        
        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)),color = colorname)
        plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate',fontsize=10)
        plt.ylabel('True Positive Rate',fontsize=10)
        plt.title('ROC Curve',fontsize=13)
        plt.legend(loc='lower right',fontsize=8)
 
    if save:
        plt.savefig('multi_models_roc.png')
        
    return plt


names = ['MobileNetV2',
         'RESnet50',
         'VGG19'
         ]
 
# sampling_methods >- y_pred.
sampling_methods = [Yscore,
                    Yscore2,
                    Yscore3
                   ]
#color:'crimson','orange','gold','mediumseagreen','steelblue', 'mediumpurple' 
colors = ['crimson',
          'orange',
          'green'
         ]
 
#ROC curves
train_roc_graph = multi_models_roc(names, sampling_methods, colors,Ytrue, save = True)
train_roc_graph.savefig('ROC_Train_all.png')
