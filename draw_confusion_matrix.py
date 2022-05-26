from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_predict = pd.read_csv('./result/VGG16-predict.csv')
data_hair_predict = np.array(data_predict.loc[:, 'hair'])
data_hair_color_predict = np.array(data_predict.loc[:, 'hair_color'])
data_gender_predict = np.array(data_predict.loc[:, 'gender'])
data_earring_predict = np.array(data_predict.loc[:, 'earring'])
data_smile_predict = np.array(data_predict.loc[:, 'smile'])
data_frontal_face_predict = np.array(data_predict.loc[:, 'frontal_face'])
data_style_predict = np.array(data_predict.loc[:, 'style'])


data_true = pd.read_csv('./result/VGG16-label.csv')
data_hair_true = np.array(data_true.loc[:, 'hair'])
data_hair_color_true = np.array(data_true.loc[:, 'hair_color'])
data_gender_true = np.array(data_true.loc[:, 'gender'])
data_earring_true = np.array(data_true.loc[:, 'earring'])
data_smile_true = np.array(data_true.loc[:, 'smile'])
data_frontal_face_true = np.array(data_true.loc[:, 'frontal_face'])
data_style_true = np.array(data_true.loc[:, 'style'])


C = confusion_matrix(data_style_true, data_style_predict, labels=[0, 1, 2])
# C = np.around(C.astype('float')/C.sum(axis=1)[:,np.newaxis],decimals=2)

plt.matshow(C, cmap=plt.cm.GnBu)
plt.colorbar()

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

plt.tick_params(labelsize=10)

plt.title('Style confusion matrix', fontdict={'family': 'Times New Roman', 'size': 15})
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 12}) # 设置字体大小。
plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 12})

# plt.savefig('./analysis/style_confusion_matrix.png')
plt.show()
