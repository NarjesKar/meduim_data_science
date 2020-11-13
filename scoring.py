import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc , roc_auc_score
import numpy as np
import itertools
from sklearn.preprocessing import scale

# plotting confusion matrix
# plitting classification report
# Feature importance analysis given by algo
# plotting ROC curves

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_feature_importance(importances ,  features_names ,thresh = None, title = None , figsize = None):   
    """
    This function plots the feature contribution to a decsion
    algo could be either a permutation result or ML algorithm
    """
    if(title is None):
        title = "Feature importance"
    if(figsize is None):
        figsize = (12, 6) 
    std = None
    if "std" in list(importances):
        std = importances["std"]
    #feature_importances["std"]
        
    indices = np.argsort(importances)[::-1]
    if(thresh is not None):
        indices = indices[:thresh]
    sorted_names = [features_names[x] for x in indices]
       
    print("Feature ranking:")
    plt.figure(figsize = figsize)
    plt.title(title)
    #if(std is not None):
    #    plt.bar(range(len(features_names)), importances[indices],color="r", yerr=std[indices], align="center")
    #else:
    plt.bar(range(len(features_names)), importances[indices],color="r", align="center")
        
    plt.xticks(range(len(features_names)), sorted_names,rotation='vertical')
    plt.xlim([-1, len(features_names)])
    plt.show()
    
def plot_class_feature_importance(features_names,result,class_names = ["No Attr", "Attr"],thresh = None):
    #N, M = X.shape
    #X = scale(X)
    #importances = feature_importances["importances"]
    #result = {}
    #for c in set(Y):
    #    result[c] = dict(zip(range(N), np.mean(X[Y==c, :], axis=0)*importances))
        
    for t, i in zip(class_names, range(len(result))):
        # sort result
        # take top n (thresh ) important features
        #sorted_results = 
        indices = np.argsort(result[i])[::-1]
        
        sorted_names = [features_names[x] for x in indices]
        
        plt.figure(figsize = (12, 6) )
     
        plt.title(t)
        
        plt.bar(range(len(result[i])), result[i][indices],color="r", align="center")
        plt.xticks(range(len(result[i])), sorted_names, rotation=90)
        plt.xlim([-1, len(result[i])])
        plt.show()

def plot_roc_curve(perfs):
    #y_pred_prob = model.predict_proba(X_test)[:,1]
    # Generate ROC curve values: fpr, tpr, thresholds
    #fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    fpr = perfs["fpr"]
    tpr = perfs["tpr"]
    thresholds = perfs["thresholds"]

    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
