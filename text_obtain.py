import os
import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_path = '/media/WD_2T/ZYJ/huggingface/tiny/'
model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
model.cuda()
config = model.config
tokenizer = AutoTokenizer.from_pretrained(hf_path, use_fast=False,
                                          model_max_length=config.tokenizer_model_max_length,
                                          padding_side=config.tokenizer_padding_side)
# prompt = "What are these?"
prompt = "What are the artificial features in this picture?"

def obtain_image(img_path):
    imgs = []
    imgs_name = []
    for root, dirs, files in os.walk(img_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith('.png'):
                imgs_name.append(file)
                imgs.append(os.path.join(root,file))
    return imgs,imgs_name

imgs_path = "/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/train/A"
imgs, imgs_name = obtain_image(imgs_path)

for i in range(len(imgs)):
    output_text, genertaion_time = model.chat(prompt=prompt, image=imgs[i], tokenizer=tokenizer)
    # 打开已存在的JSON文件，如果不存在则创建一个新的文件
    with open("/media/chill/PortableSSD/查违数据/to_zyj7.10/alighed_256_8_1_1_red/train/A.json", "r+", encoding="utf-8") as file:
        # 尝试读取文件中的现有数据
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            # 如果文件为空或不是有效的JSON格式，创建一个空字典
            data = {}

        # 向字典中添加新的键值对
        data[imgs[i]] = output_text

        # 将文件指针移回文件开头
        file.seek(0)
        # 清空文件内容
        file.truncate()
        # 将更新后的字典写回JSON文件
        json.dump(data, file, ensure_ascii=False, indent=4)

