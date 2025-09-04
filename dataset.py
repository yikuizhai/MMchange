import cv2
import numpy
import torch.utils.data
import json
import torch
from CLIP import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''

    def __init__(self, dataset, file_root='data/', transform=None):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        self.file_list = open(file_root + '/' + dataset + '/list/' + dataset + '.txt').read().splitlines()
        self.pre_images = [file_root + '/' + dataset + '/A/' + x for x in self.file_list]
        self.post_images = [file_root + '/' + dataset + '/B/' + x for x in self.file_list]
        self.gts = [file_root + '/' + dataset + '/label/' + x for x in self.file_list]
        self.transform = transform

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_name = self.pre_images[idx]
        label_name = self.gts[idx]
        post_image_name = self.post_images[idx]
        pre_image = cv2.imread(pre_image_name)
        label = cv2.imread(label_name, 0)
        post_image = cv2.imread(post_image_name)
        img = numpy.concatenate((pre_image, post_image), axis=2)
        # if self.transform:
        #     [pre_image, label, post_image] = self.transform(pre_image, label, post_image)
        #
        # return pre_image, label, post_image

        if self.transform:
            [img, label] = self.transform(img, label)
        # # 读取json文件
        # with open('/media/WD_2T/ZYJ/315_cd_data/LEVIR_256/256/total.json', 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        # dictionary = dict(data)
        # output_text_A = dictionary.get(pre_image_name)
        # output_text_B = dictionary.get(post_image_name)
        # model_text_decode, preprocess = clip.load("ViT-B/32", device=device)
        # text_A = clip.tokenize(output_text_A).to(device)
        # text_B = clip.tokenize(output_text_B).to(device)
        # text_A_features = model_text_decode.encode_text(text_A)
        # text_B_features = model_text_decode.encode_text(text_B)

        return img, label,pre_image_name,post_image_name

    def get_img_info(self, idx):
        img = cv2.imread(self.pre_images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}
