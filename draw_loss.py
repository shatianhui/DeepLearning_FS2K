import matplotlib.pyplot as plt
import pandas as pd


data_loss = pd.read_csv('./result/VGG16-losses.csv')
plt.plot(data_loss.iloc[:, 0], data_loss.iloc[:, 1], color="blue", label="loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
