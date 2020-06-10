# Fingernails-segmentation-by-Unet
使用Unet语义分割网络实现对指甲盖的分割
思路如下：
Unet在医疗图像中优秀的分割性能，依据此我在数据集的准备中采集的是同样只有少量语义干扰的黑色/白色背景 手部平放在正中间，1：1图像大小的数据集，目前网上并找不到相关数据集，数据都是采用labelme标注的，采集了大概200张数据然后经过旋转和少量平移的数据增强最终是1500的数据大小。在gpu下每一张图片的predict速度为0.02s。

https://github.com/Golbstein/Fingernails-Segmentation
本项目的原始代码在此之上进行改进，主要做了数据集之上的处理。
目前的检测效果在纯色背景下效果更加显著。

