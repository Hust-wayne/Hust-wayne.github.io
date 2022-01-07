---
layout:     post
title:      OpenVino --- C++ 推理流程
subtitle:   ----
date:       2022-01-07
author:     wayne
header-img: img/the-first.png
catalog: false
tags:
    - C++推理


---

- **openvino 环境配置相关**
- **C++案例流程**

***

### openvino环境匹配

```
1. openvino 2021.3.394
2. vs2019
3. cmake3.20.2
4. python 3.6-3.9 (用于转换其他框架模型到IR, 也支持python推理)

```

##### 系统环境说明（永久性配置可参考setupvars.bat中信息，配置后就不必执行setupvars.bat初始环境）

​	 安装好openvino后主要配置好系统环境变量ngraph_DIR、TBB_DIR、OPENCV_DIR（openvino自带opencv)等，这里配置个人觉得用处不大，放便cmake工具吧？主要还是将Path配置好即可；

| 变量名              | 变量值                                                       |
| ------------------- | ------------------------------------------------------------ |
| INTEL_OPENVINO_DIR  | C:\Program Files (x86)\Intel\openvino_2021.3.394\bin         |
| HDDL_INSTALL_DIR    | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\hddl |
| InferenceEngine_DIR | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\share |
| OpenCV_DIR          | %INTEL_OPENVINO_DIR%\opencv\cmake                            |
| ngraph_DIR          | %INTEL_OPENVINO_DIR%\deployment_tools\ngraph\cmake           |
| TBB_DIR             | %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\cmake |

​	系统环境Path: 上面的系统环境变量与Path的配置关系不大，唯一可通过***%XXX%\XXX***引入需要的环境（简洁，占字节少），此版本openvino需要配置如下信息：

| %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Release |
| :----------------------------------------------------------- |
| %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\bin\intel64\Debug |
| %INTEL_OPENVINO_DIR%\opencv\bin                              |
| %INTEL_OPENVINO_DIR%\deployment_tools\inference_engine\external\tbb\bin |
| %INTEL_OPENVINO_DIR%\deployment_tools\ngraph\lib             |

​	*说明：上面的Path环境需要全部配置，否则在python环境中--import cv2--(使用自带的opencv--cv2.pyd)会找不到dlll错误*

##### python环境说明

​	单独创建一个独立的python虚拟环境，直接将openvino自带的一些环境(根据自己使用的python环境全部拷贝过去就行)复制到site-packages中，不用按照网上所述的需要pip安装一些openvino包，也不需要添加***"PYTHONPATH"***系统环境变量。

<img src="/imgs_f_md/openvnio/Snipaste_2021-05-31_13-40-01.jpg" style="zoom:50%;" />

不用担心相关文件找不到，python 导包的搜索顺序：

```
1. 当前目录
2. 环境变量PYTHONPATH中的目录
3. Python安装目录第三方库（for Linux OS：/usr/local/lib/python）（win: site-packages下）
```

***

#### C++推理流程

将Inter官方提供的person-vehicle-bike-detection-2002为例子、

![](/imgs_f_md/openvnio/person-vehicle-bike-detection-2002.png)

![](/imgs_f_md/openvnio/Snipaste_2021-05-31_13-49-40.jpg)

这里将使用官方提供的FP32模型，上图是模型需要的输入以及输出结果。

```c++
#include <inference_engine.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include "func.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace InferenceEngine;

void test_demo() {

	// crerate IE engine, find supported device name
	Core ie;
	vector<string> availableDevices = ie.GetAvailableDevices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  load model
	auto network = ie.ReadNetwork("D:/download/intel/person-vehicle-bike-detection-2002/FP32/person-vehicle-		bike-detection-2002.xml", "D:/download/intel/person-vehicle-bike-detection-2002/FP32/person-vehicle-		bike-detection-2002.bin");

	// request network input and output info
	InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
	InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());

	// set input format
	for (auto& item : input_info) {
		auto input_data = item.second;
		input_data->setPrecision(Precision::FP32);
		input_data->setLayout(Layout::NCHW);
		input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
	}

	// set output format
	for (auto& item : output_info) {
		auto output_data = item.second;
		output_data->setPrecision(Precision::FP32);
	}
	auto executable_network = ie.LoadNetwork(network, "GPU");

	// output result 
	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;

	// request infer
	auto infer_request = executable_network.CreateInferRequest();

	Mat src = imread("F:/Downloads/opencv_tutorial_data/images/objects.jpg");
	int image_height = src.rows;
	int image_width = src.cols;

	/** Iterating over all input blobs **/
	for (auto& item : input_info) {
        auto input_name = item.first;

        /** Getting input blob **/
        auto input = infer_request.GetBlob(input_name);
        size_t num_channels = input->getTensorDesc().getDims()[1];
        size_t h = input->getTensorDesc().getDims()[2];
        size_t w = input->getTensorDesc().getDims()[3];
        size_t image_size = h * w;
        Mat blob_image;
        resize(src, blob_image, Size(w, h));
        cvtColor(blob_image, blob_image, COLOR_BGR2RGB);

        // NCHW
        float* data = static_cast<float*>(input->buffer());
        for (size_t row = 0; row < h; row++) {
            for (size_t col = 0; col < w; col++) {
                for (size_t ch = 0; ch < num_channels; ch++) {
                    data[image_size * ch + row * w + col] = float(blob_image.at<Vec3b>(row, col)[ch]);
                }
            }
        }
	}

	int64 start = getTickCount();

	// do inference
	infer_request.Infer();

	for (auto& item : output_info) {
		auto output_name = item.first;
		printf("output_name : %s \n", output_name.c_str());
		// get output blob
		auto output = infer_request.GetBlob(output_name);

		const float* output_blob = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output-											>buffer());
		const SizeVector outputDims = output->getTensorDesc().getDims();

		const int out_num = outputDims[2];   // 200
		const int out_info = outputDims[3];  // 7

		for (int n = 0; n < out_num; n++) {
			float conf = output_blob[n * 7 + 2];
			if (conf < 0.5) {
				continue;
			}
			int x1 = saturate_cast<int>(output_blob[n * 7 + 3] * image_width);
			int y1 = saturate_cast<int>(output_blob[n * 7 + 4] * image_height);
			int x2 = saturate_cast<int>(output_blob[n * 7 + 5] * image_width);
			int y2 = saturate_cast<int>(output_blob[n * 7 + 6] * image_height);

			//label
			int label = saturate_cast<int>(output_blob[n * 7 + 1]);

			classIds.push_back(label);
			confidences.push_back(conf);
			boxes.push_back(Rect(x1, y1, x2 - x1, y2 - y1));
		}
	}

	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.25, 0.5, indices);
	float time = (getTickCount() - start) / getTickFrequency();

	printf("time used：%f", time);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		rectangle(src, box, Scalar(140, 199, 0), 4, 8, 0);
	}

	imshow("OpenVINO-test", src);
	waitKey(0);
}
```

![](/imgs_f_md/openvnio/Snipaste_2021-05-31_14-25-28.jpg)

**最终结果如上图所示，推理+后处理一般在30多ms,  在Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz   2.59 GHz上**

