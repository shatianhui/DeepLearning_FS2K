import json
from torchvision import transforms

# 解析json文件
def Dejson(selected_attrs, json_path):
    fp = open(json_path, 'r')
    data = json.load(fp)
    img_path_list = list()
    attrs_list = list()
    for item in data:
        str = item['image_name'].replace('photo', 'sketch').replace('image', 'sketch')  # 获得图片名称
        str += '.jpg' if (
                '1' in item['image_name'].split('/')[0] or '3' in item['image_name'].split('/')[0]) else '.png'  # 图片的后缀
        img_path_list.append(str)
        attrs = list()
        for attr in selected_attrs:
            attrs.append(item[attr])
        attrs_list.append(attrs)
    return img_path_list, attrs_list


# 将图片变成正方形 -> 变为tensor变量 -> 进行标准化
def set_transform():
    transform=transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    ])
    return transform


