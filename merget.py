import json

# # 定义要合并的文件名列表
file_names = ['/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/train/A.json', '/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/train/B.json', '/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/test/A.json', '/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/test/B.json', '/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/val/A.json', '/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/val/B.json']
#
# 初始化一个空列表来存储所有文件的数据
merged_dict = {}

# 遍历文件名列表，逐个读取并合并数据
for file_name in file_names:
    with open(file_name, 'r') as file:
        data = json.load(file)
        merged_dict.update(data)

# 将合并后的数据写入新的JSON文件
with open('chawei.json', 'w') as file:
    json.dump(merged_dict, file)


# with open('/media/WD_2T/ZYJ/huggingface/CLIP/levir_plus.json','r',encoding='utf-8') as file:
#     data = json.load(file)
#
# dictionary = dict(data)
# print(dictionary.get('/media/WD_2T/ZYJ/315_cd_data/LEVIR_256/256/train/B/train_224_1.png'))
