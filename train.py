import torch.nn.functional as F
import torch
import torch.optim as optim
import json
import copy
import pandas as pd
import numpy as np

import config as cfg
from model import VGGTransferModel
from utils import set_transform
from FS2K import get_loader


def main():
    epochs = cfg.epochs  # 迭代次数
    batch_size = cfg.batch_size  # 批处理大小
    learning_rate = cfg.lr  # 学习率
    selected_attrs = cfg.selected_attrs  # 分类的属性 头发|笑容｜性别..
    json_train_path = cfg.json_train_path  # train_json路径 来加载训练数据集
    json_test_path = cfg.json_test_path  # test_json路径 来加载测试数据集
    device = torch.device("cuda:" + str(cfg.DEVICE_ID) if torch.cuda.is_available() else "cpu")  # 计算设备
    transform = set_transform()  # 图片预处理
    train_loader = get_loader(json_train_path, selected_attrs, batch_size, 'train', transform)  # 训练数据加载器
    test_loader = get_loader(json_test_path, selected_attrs, batch_size, 'test', transform)  # 测试数据加载器
    model_type = "VGG16"
    model = VGGTransferModel(model_type, pretrained=True).to(device)   # 获得模型
    optimer = optim.Adam(model.parameters(), lr=learning_rate)  # 优化器 负责更新参数
    scheduler = optim.lr_scheduler.MultiStepLR(optimer, [30, 80], gamma=0.1)  # 调节学习率

    train_losses = []  # 记录训练过程的总损失
    eval_acc_dict = {}  # 统计每个epoch的正确率
    best_acc = 0.0

    # 记录最好的模型以及正确率
    best_model_wts = copy.deepcopy(model.state_dict())
    for attr in selected_attrs:
        eval_acc_dict[attr] = []
    for epoch in range(epochs):
        model.train()  # 开始训练
        temp_loss = 0  # 记录这个epoch的loss
        batch_idx = 0
        for batch_idx, data in enumerate(train_loader):
            images, labels = data  # 得到图片以及label
            images = images.to(device)  # 将图片放到显卡上进行计算
            hair, hair_color, gender, earring, smile, frontal, style = model(images)  # 得到模型出来的结果
            # 计算各个属性的交叉熵损失

            hair_loss = F.cross_entropy(input=hair, target=labels[0].to(device), weight=torch.tensor([0.05, 0.95]).to(device))
            hair_color_loss = F.cross_entropy(hair_color, target=labels[1].to(device))
            gender_loss = F.cross_entropy(gender, labels[2].to(device))
            earring_loss = F.cross_entropy(earring, labels[3].to(device), weight=torch.tensor([0.9, 0.1]).to(device))
            smile_loss = F.cross_entropy(smile, labels[4].to(device))
            frontal_loss = F.cross_entropy(frontal, labels[5].to(device), weight=torch.tensor([0.1, 0.9]).to(device))
            style_loss = F.cross_entropy(style, labels[6].to(device))
            # 总共的Loss
            total_loss = hair_loss + 0.01*hair_color_loss + gender_loss + earring_loss + smile_loss + frontal_loss + style_loss
            total_loss.backward()  # 误差回传
            optimer.step()  # 更新参数
            optimer.zero_grad()  # 梯度归零
            temp_loss += total_loss.item()  # 累加这个batch的loss
            # 打印loss信息
            if (batch_idx + 1) % (len(train_loader) // 4) == 0:
                print("Epoch: %d/%d, training batch_idx:%d , loss: %.4f" % (
                    epoch, epochs, batch_idx + 1,  total_loss.item()))
        running_loss = temp_loss / (batch_idx + 1)  #  返回epoch_loss
        if epoch > epochs // 2:
            scheduler.step()
        print("Epoch: %d,  loss: %.4f , lr:%.7f" % (epoch, running_loss, learning_rate))

        # 在测试集上评估性能
        model.eval()
        correct_dict = {}  # 统计正确率
        predict_dict = {}  # 保存预测值
        label_dict = {}  # 保存label
        for attr in selected_attrs:
            correct_dict[attr] = 0  # 记录各个属性标签正确的数量
            predict_dict[attr] = list()
            label_dict[attr] = list()

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                images, labels = data
                images = images.to(device)
                hair, hair_color, gender, earring, smile, frontal, style = model(images)
                out_dict = {'hair': hair, 'hair_color': hair_color, 'gender': gender, 'earring': earring,
                            'smile': smile, 'frontal_face': frontal,
                            'style': style}
                batch = len(out_dict['hair'])
                for i in range(batch):
                    for attr_idx, attr in enumerate(selected_attrs):
                        pred = np.argmax(out_dict[attr][i].data.cpu().numpy())  # 得到预测值
                        true_label = labels[attr_idx].data.cpu().numpy()[i]  # 得到label
                        if pred == true_label:
                            correct_dict[attr] = correct_dict[attr] + 1
                        predict_dict[attr].append(pred)
                        label_dict[attr].append(true_label)
        mAP = 0  # 存放各个属性分类精度的平均值
        for attr in selected_attrs:
            correct_dict[attr] = correct_dict[attr] * 100 / (len(test_loader) * batch_size)
            mAP += correct_dict[attr]
        mAP /= len(selected_attrs)
        print("Epoch: {} accuracy:{}".format(epoch, correct_dict))
        print("Epoch: {} mAP: {}".format(epoch, mAP))
        train_losses.append(running_loss)
        for attr in selected_attrs:
            eval_acc_dict[attr].append(correct_dict[attr])

        # 比较正确率并保存最佳模型
        if mAP > best_acc:
            best_acc = mAP
            best_model_wts = copy.deepcopy(model.state_dict())
            best_predict_dict = predict_dict
            best_label_dict = label_dict

    # 保存每个epoch的每个属性的正确率
    eval_acc_csv = pd.DataFrame(eval_acc_dict, index=[i for i in range(epochs)]).T
    eval_acc_csv.to_csv("./result/" + model_type + "-eval_accuracy" + ".csv")
    # 保存训练过程的loss
    train_losses_csv = pd.DataFrame(train_losses)
    train_losses_csv.to_csv("./result/" + model_type + "-losses" + ".csv")
    # 保存best model
    model_save_path = "./result/" + model_type + "-best_model_params" + ".pth"
    torch.save(best_model_wts, model_save_path)
    print("The model has saved in {}".format(model_save_path))
    # 保存预测值
    pred_csv = pd.DataFrame(best_predict_dict)
    pred_csv.to_csv("./result/" + model_type + "-predict" + ".csv")
    # 保存真实值
    label_csv = pd.DataFrame(best_label_dict)
    label_csv.to_csv("./result/" + model_type + "-label" + ".csv")
    # 保存模型信息
    report_dict = {}
    report_dict["model"] = model_type
    report_dict["best_mAP"] = best_acc
    report_dict["lr"] = learning_rate
    report_dict["optim"] = 'Adam'
    report_dict['Batch_size'] = batch_size
    report_json = json.dumps(report_dict)
    report_file = open("./result/" + model_type + "-report.json", 'w')
    report_file.write(report_json)
    report_file.close()
    print("训练完成")

if __name__ == '__main__':
    main()

