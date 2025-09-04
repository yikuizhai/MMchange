# MMchange
# <p align=center>`Multimodal Feature Fusion Network with Text Difference Enhancement for Remote Sensing Change Detection`</p>

> **Authors:**
Yijun Zhou, Yikui Zhai, Zilu Ying, Tingfeng Xian, Wenlve Zhou, Zhiheng Zhou, Xiaolin Tian, Xudong Jia, Hongsheng Zhang,  C. L. Philip Chen


### 1. Abstract
Although deep learning has advanced remote sensing change detection (RSCD), most methods rely solely on image modality, limiting feature representation, change pattern modeling, and generalization—especially under illumination and noise disturbances. To address this, we propose MMChange, a multimodal RSCD method that combines image and text modalities to enhance accuracy and robustness. An Image Feature Refinement (IFR) module is introduced to highlight key regions and suppress environmental noise. To overcome the semantic limitations of image features, we employ a vision-language model (VLM) to generate semantic descriptions of bi-temporal images. A Textual Difference Enhancement (TDE) module then captures fine-grained semantic shifts, guiding the model toward meaningful changes. To bridge the heterogeneity between modalities, we design an Image-Text Feature Fusion (ITFF) module that enables deep cross-modal integration. Extensive experiments on LEVIR-CD, WHU-CD, and SYSU-CD demonstrate that MMChange consistently surpasses state-of-the-art methods across multiple metrics, validating its effectiveness for multimodal RSCD. Code is available at: https://github.com/yikuizhai/MMChange.

### 2. Overview


<p align="center">
    <img width="1068" height="417" alt="image" src="https://github.com/user-attachments/assets/2e846ff2-7f39-424b-9c3c-a3cb4db18046" />
 <br />
</p>

### 2. Usage
+ Prepare the data:
    - Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/), [BCDD](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html), and [SYSU](https://github.com/liumency/SYSU-CD)
    - Crop LEVIR and BCDD datasets into 256x256 patches. The pre-processed BCDD dataset can be obtained from [BCDD_256x256](https://drive.google.com/file/d/1VrdQ-rxoGVM_8ecA-ObO0u-O8rSTpSHA/view?usp=sharing).
    - Generate list file as `ls -R ./label/* > test.txt`
    - Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

+ Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n A2Net python=3.8`
    - Installing necessary packages: `pip install -r requirements.txt `

