1.需要预先将requirement中的modules安装，
2.之后将下载的数据集解压在data文件夹下
3.使用tools中的文件预处理数据集，主要是进行数据集的划分以及图像的resize
4.使用inference进行模型的测试输出对应测预测结果文件。
5.model_a_00,a_01,a_02分别使用了不同的encoder(efficientnet, resnet34, resnext)
