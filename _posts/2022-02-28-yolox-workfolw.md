---
layout:     post
title:      workflow-yolox
subtitle:   yolox训练流程
date:       2022-02-07
author:     wayne
header-img: img/the-first.png
catalog: false
tags:
    - DL训练工作流



---

- **YOLOX工作流程**

• [code](https://github.com/Megvii-BaseDetection/YOLOX)     •  [paper](https://arxiv.org/abs/2107.08430)

YOLOX 是旷视开源的高性能检测器。旷视的研究者将解耦头、数据增强、anchor-free以及标签分类等目标检测领域的优秀进展与 YOLO 进行了巧妙的集成组合，提出了 YOLOX，不仅实现了超越 YOLOv3、YOLOv4 和 YOLOv5 的 AP，而且取得了极具竞争力的推理速度。如下图：

![](/imgs_f_md/yolox/1.png)

其中YOLOX-L版本以 68.9 FPS 的速度在 COCO 上实现了 50.0% AP，比 YOLOv5-L 高出 1.8% AP！还提供了支持 ONNX、TensorRT、NCNN 和 Openvino 的部署版本，本文将详细介绍如何使用 YOLOX进行物体检测。

## 一、 配置环境

本机的环境：

| systerm     | ubantu16.04 |
| ----------- | ----------- |
| Pytorch版本 | 1.8.0       |
| Cuda版本    | 11.1        |
|             |             |

1)安装YOLOX

```bash
conda create -n yolox python=3.8
conda activate yolox
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip install -U pip && pip install -r requirements.txt
pip install -v -e .  # or  python setup.py develop
```

2)安装apex

```bash
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--pyprof" --global-option="--cpp_ext" --global-option="--cuda_ext" ./
python setup.py install --cpp_ext --cuda_ext
```

3)安装pycocotools

```sh
pip install cython
pip install pycocotools
# 不成功使用 pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```



## 二、 制作数据集

数据集采用VOC格式数据集，原始数据集是Labelme标注的数据集（json）。

```python
# labelme(json2voc)
import os
import numpy as np
import codecs
import json
from glob import glob
import cv2
import shutil
from sklearn.model_selection import train_test_split
#1.标签路径
labelme_path = "./"              #原始labelme标注数据路径
saved_path = "./VOC2007/"                #保存路径

#2.创建要求文件夹
if not os.path.exists(saved_path + "Annotations"):
    os.makedirs(saved_path + "Annotations")
if not os.path.exists(saved_path + "JPEGImages/"):
    os.makedirs(saved_path + "JPEGImages/")
if not os.path.exists(saved_path + "ImageSets/Main/"):
    os.makedirs(saved_path + "ImageSets/Main/")
    
#3.获取待处理文件
files = glob(labelme_path + "*.json")
files = [i.split("/")[-1].split(".json")[0] for i in files]

#4.读取标注信息并写入 xml
for json_file_ in files:
    json_filename = labelme_path + json_file_ + ".json"
    json_file = json.load(open(json_filename,"r",encoding="utf-8"))
    height, width, channels = cv2.imread(labelme_path + json_file_ +".jpg").shape
    with codecs.open(saved_path + "Annotations/"+json_file_ + ".xml","w","utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>' + 'UAV_data' + '</folder>\n')
        xml.write('\t<filename>' + json_file_ + ".jpg" + '</filename>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>The UAV autolanding</database>\n')
        xml.write('\t\t<annotation>UAV AutoLanding</annotation>\n')
        xml.write('\t\t<image>flickr</image>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t</source>\n')
        xml.write('\t<owner>\n')
        xml.write('\t\t<flickrid>NULL</flickrid>\n')
        xml.write('\t\t<name>ChaojieZhu</name>\n')
        xml.write('\t</owner>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+ str(width) + '</width>\n')
        xml.write('\t\t<height>'+ str(height) + '</height>\n')
        xml.write('\t\t<depth>' + str(channels) + '</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t\t<segmented>0</segmented>\n')
        for multi in json_file["shapes"]:
            points = np.array(multi["points"])
            xmin = min(points[:,0])
            xmax = max(points[:,0])
            ymin = min(points[:,1])
            ymax = max(points[:,1])
            label = multi["label"]
            if xmax <= xmin:
                pass
            elif ymax <= ymin:
                pass
            else:
                xml.write('\t<object>\n')
                xml.write('\t\t<name>'+"class_name"+'</name>\n') 
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>1</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
                xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
                xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
                xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
                print(json_filename,xmin,ymin,xmax,ymax,label)
        xml.write('</annotation>')
        
#5.复制图片到 VOC2007/JPEGImages/下
image_files = glob(labelme_path + "*.jpg")
print("copy image files to VOC007/JPEGImages/")
for image in image_files:
    shutil.copy(image,saved_path +"JPEGImages/")
    
#6.split files for txt
txtsavepath = saved_path + "ImageSets/Main/"
ftrainval = open(txtsavepath+'/trainval.txt', 'w')
ftest = open(txtsavepath+'/test.txt', 'w')
ftrain = open(txtsavepath+'/train.txt', 'w')
fval = open(txtsavepath+'/val.txt', 'w')
total_files = glob("./VOC2007/Annotations/*.xml")
total_files = [i.split("/")[-1].split(".xml")[0] for i in total_files]
#test_filepath = ""
for file in total_files:
    ftrainval.write(file + "\n")
#test
#for file in os.listdir(test_filepath):
#    ftest.write(file.split(".jpg")[0] + "\n")
#split
train_files,val_files = train_test_split(total_files,test_size=0.15,random_state=42)
#train
for file in train_files:
    ftrain.write(file + "\n")
#val
for file in val_files:
    fval.write(file + "\n")

ftrainval.close()
ftrain.close()
fval.close()
#ftest.close()

```

```bash
|-- datasets
|   |-- VOCdevkit
|   |   |-- VOC2007
|   |   |   |-- Annotations
|   |   |   |-- ImageSets
|   |   |   |   -- Main
|   |   |   |-- JPEGImages

```



## 三、 修改数据配置文件

### **1.  修改类别**

文件路径：**exps/example/yolox_voc/yolox_voc_s.py**，本次使用的类别有2类，所以将**self.num_classes**修改为2。

```python
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 2
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
```

打开**yolox/data/datasets/voc_classes.py**文件，修改为自己的类别名：

```python
VOC_CLASSES = (
    "fire",
    "smoke",
)
```

**2. 修改数据集目录**

文件路径：exps/example/yolox_voc/yolox_voc_s.py，修改**data_dir**及**image_sets**，最终结果如下：

```python
dataset = VOCDetection(
                data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
                image_sets=[('2007', 'trainval')],
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

 valdataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "VOCdevkit"),
            image_sets=[('2007', 'test')],
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

```

重新编译yolox

```bash
python setup.py install
```



## 四、 训练

执行命令训练：

```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 --fp16  -c yolox_s.pth
```



## 五、 测试

**5.1 单张图片预测**

使用训练好的模型进行测试。测试调用tools/demo.py,先用命令行的方式演示：

```python
python tools/demo.py image -f exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth --path ./assets/aircraft_107.jpg --conf 0.3 --nms 0.65 --tsize 640 --save_result --device gpu
```



## 六. onnx inference

| cuda | onnxruntime |
| :--: | :---------: |
| 10.+ |     1.4     |
| 11.+ |    1.9+     |

1.  执行命令转换模型到.onnx:

   ```bash
   python tools/export_onnx.py --output-name YOLOX_outputs/firesmoke_s/yolox_s.onnx -f  exps/example/yolox_voc/yolox_voc_s.py -c YOLOX_outputs/yolox_voc_s/best_ckpt.pth
   ```



## 七. 遇到的错误

### **1、RuntimeError: DataLoader worker (pid(s) 9368, 12520, 6392, 7384) exited unexpectedly**

错误原因：torch.utils.data.DataLoader中的num_workers错误 将num_workers改为0即可，0是默认值。num_workers是用来指定开多进程的数量，默认值为0，表示不启用多进程。

打开yolox/exp/yolox_base.py,将**self.data_num_workers**设置为0，如下图：

将num_workers设置为0，程序报错，并提示设置环境变量KMP_DUPLICATE_LIB_OK=TRUE 那你可以在设置环境变量KMP_DUPLICATE_LIB_OK=TRUE 或者使用临时环境变量：（在代码开始处添加这行代码)

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

**2、RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR****或显存溢出等**

执行命令

```bash
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 --fp16 -o -c yolox_s.pth
```

把“-o”去掉后就正常了。

```
python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 --fp16  -c yolox_s.pth
```

**3、训练voc时eval，ap, map为0**

- 统一 image（.jpg）,	label(.xml)格式，禁止（.JPG .XML）
- 规范数据集命名，名称不要含空格等，会导致eval代码报错或ap=0
