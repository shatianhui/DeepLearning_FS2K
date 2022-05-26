root = 'FS2K/sketch'  # 图像的根目录
selected_attrs = ['hair', 'hair_color', 'gender', 'earring', 'smile', 'frontal_face', 'style']  # 选择的属性  有无头发，头发颜色，性别，有无耳环，微笑，正面脸 ，风格
json_train_path = 'FS2K/anno_train.json'
json_test_path = 'FS2K/anno_test.json'

# 可能需要更改的参数
DEVICE_ID = '0'
epochs = 50
batch_size = 16
lr = 1e-5

