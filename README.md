# 对FS2K进行属性分类
## 1 简介
* 数据集FS2K下载地址：https://github.com/DengPingFan/FS2K
* 使用VGG16,去掉网络最后的全连接层对输入的素描图提取特征，自定义一个全连接层fc作为分类器，将得到的图像特征输入到多个全连接层fc，不同的fc对应不同的分类任务。
## 2 环境配置
* Python 3.7
* Pytorch1.10.2
* 使用GPU训练
## 3 文件结构
```
  ├── FS2K: 存放数据集和测试集
  │    ├── photo: 人脸彩色图
  │    ├── sketch: photo文件夹里对应的人脸素描图
  │    ├── test: 测试集
  │    ├── train: 训练集
  │    ├── anno_test.json: 测试集对应的标签json文件
  │    └── anno_train.json: 训练集对应的标签json文件
  │ 
  ├── tools: 工具类
  │    ├── check.py: 训练集和测试集各属性各类别的数量
  │    ├── split_train_test.py: 将数据划分为训练集和测试集
  │    └── vis.py: 通过opencv将数据可视化
  │
  ├── config.py: 定义路径和可能需要更改的参数，比如学习率等
  ├── draw_confusion_matrix.py: 绘制混淆矩阵
  ├── model.py: 模型搭建文件
  ├── train.py: 在训练集上进行训练，在测试集评估，保存在测试集上精度最高的模型
  ├── FS2K.py: 加载数据集
  └── utils.py: 工具类，解析json文件和对图片进行预处理
```

## 4 使用方法
* 下载数据集[FS2K](https://github.com/DengPingFan/FS2K)
* 运行split_train_test.py，划分训练集和测试集
* 运行train.py,结果保存在result文件夹中
## 5 结果
| **attr** | hair   | hair_color | gender | earring | smile  | frontal_face | style  |
|----------| ------ | ---------- | ------ | ------- | ------ | ------------ | ------ |
| **acc**  | 0.9510 | 0.4510     | 0.9154 | 0.7846  | 0.7663 | 0.8779       | 0.9144 |

