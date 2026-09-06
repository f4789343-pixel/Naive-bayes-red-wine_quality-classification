import matplotlib.pyplot as plt
from scikit_learn import cm, y_test
import numpy as np

classes = np.unique(y_test)
plt.imshow(cm)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive bayes')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)

for i in range(len(classes)):
  for j in range(len(classes)):
    plt.text(j, i, cm[i][j], ha='center',va='center')
plt.colorbar()
plt.savefig('Confusion_Matrix.png')
plt.show()