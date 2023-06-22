from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

models_name = ['svc_linear','svc_poly', 'svc_rbf', 'linear_svc', 'dt', 'rf', 'knn', 'mlp']
results = [(0.9275,0.9171), (0.9439, 0.8961), (0.9353,0.9018), (0.9353,0.90), (0.9207,0.9064), (0.9445,0.8669), (0.91,0.8686), (0.8245, 0.864)]

true = [0 for i in range(102400)]
true.extend([1 for j in range(102400)])

for i in range(8):

    model_name = models_name[i]

    recall_0 = results[i][0]
    recall_1 = results[i][1]

    t0 = round(recall_0 * 102400)
    t1 = round(recall_1 * 102400)

    prediction = [0 for i in range(t0)]
    prediction.extend([1 for i in range(102400 - t0)])
    prediction.extend([1 for i in range(t1)])
    prediction.extend([0 for i in range(102400-t1)])

    confusion_matrix = ConfusionMatrixDisplay.from_predictions(true, prediction,
                                                               display_labels=("NoEngagement", "Engagement"),
                                                               values_format='d')#, normalize='all')

    plt.tight_layout(pad=3)

    plt.savefig(f"confusion_matrix_{model_name}.png")

    plt.close()
