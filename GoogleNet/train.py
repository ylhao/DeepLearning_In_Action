#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: UTF-8 -*-
"""
训练常用视觉基础网络，用于分类任务
需要将训练图片，类别文件 label_list.txt 放置在同一个文件夹下
程序会先读取 train.txt 文件获取类别数和图片数量
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import math
import paddle
import paddle.fluid as fluid
import codecs
import logging

from paddle.fluid.initializer import MSRA
from paddle.fluid.initializer import Uniform
from paddle.fluid.param_attr import ParamAttr
from PIL import Image
from PIL import ImageEnhance


# In[2]:


train_parameters = {   
    "input_size": [3, 224, 224],
    "class_dim": -1,  # 分类数，会在初始化自定义 reader 的时候获得   
    "image_count": -1,  # 训练图片数量，会在初始化自定义 reader 的时候获得   
    "label_dict": {},  # 类别编号与类别名之间的映射关系
    "data_dir": "data/data2815",  # 训练数据存储地址   
    "train_file_list": "train.txt",   
    "label_file": "label_list.txt",   
    "save_freeze_dir": "./freeze-model",   
    "save_persistable_dir": "./persistable-params",   
    "continue_train": False,        # 是否接着上一次保存的参数接着训练，优先级高于预训练模型   
    "pretrained": True,            # 是否使用预训练的模型   
    "pretrained_dir": "data/GoogleNet_pretrained",    
    "mode": "train",   
    "num_epochs": 120,   
    "train_batch_size": 30,   
    "mean_rgb": [127.5, 127.5, 127.5],  # 常用图片的三通道均值，通常来说需要先对训练数据做统计，此处仅取中间值   
    "use_gpu": True,   
    "dropout_seed": None,   
    "image_enhance_strategy": {  # 图像增强相关策略   
        "need_distort": True,  # 是否启用图像颜色增强   
        "need_rotate": True,   # 是否需要增加随机角度   
        "need_crop": True,      # 是否要增加裁剪   
        "need_flip": True,      # 是否要增加水平随机翻转   
        "hue_prob": 0.5,   
        "hue_delta": 18,   
        "contrast_prob": 0.5,   
        "contrast_delta": 0.5,   
        "saturation_prob": 0.5,   
        "saturation_delta": 0.5,   
        "brightness_prob": 0.5,   
        "brightness_delta": 0.125   
    },   
    "early_stop": {   
        "sample_frequency": 50,   
        "successive_limit": 3,   
        "good_acc1": 0.92   
    },   
    "rsm_strategy": {   
        "learning_rate": 0.001,   
        "lr_epochs": [20, 40, 60, 80, 100],   
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]   
    },   
    "momentum_strategy": {   
        "learning_rate": 0.001,   
        "lr_epochs": [20, 40, 60, 80, 100],   
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]   
    },   
    "sgd_strategy": {   
        "learning_rate": 0.001,   
        "lr_epochs": [20, 40, 60, 80, 100],   
        "lr_decay": [1, 0.5, 0.25, 0.1, 0.01, 0.002]   
    },   
    "adam_strategy": {   
        "learning_rate": 0.002   
    }   
}   


# In[3]:


class GoogleNet():
    """
    GoogleNet网络类
    """
    def __init__(self):
        self.params = train_parameters

    def conv_layer(self,
                   input,
                   num_filters,  # 卷积核数量
                   filter_size,  # 卷积核尺寸
                   stride=1,  # 步长
                   groups=1,  
                   act=None,
                   name=None):
        channels = input.shape[1]  # 通道数
        # 卷积层权重初始化方式,随机均匀初始化
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,  # 卷积核数量
            filter_size=filter_size,  # 卷积核尺寸
            stride=stride,  # 步长
            padding=(filter_size - 1) // 2,  # 填充大小
            groups=groups,  # Conv2d转置层的groups个数
            act=act,  # 激活函数类型
            param_attr=param_attr,
            bias_attr=False,  # bias
            name=name)
        return conv

    # xavier 初始化
    def xavier(self, channels, filter_size, name):
        stdv = (3.0 / (filter_size**2 * channels))**0.5
        param_attr = ParamAttr(
            initializer=fluid.initializer.Uniform(-stdv, stdv),
            name=name + "_weights")
        return param_attr

    # 定义 inception 结构
    def inception(self,
                  input,
                  channels,
                  filter1,  # 1维卷积核的数量
                  filter3R,  # 3 * 3 卷积核前置 1 * 1 卷积核的数量
                  filter3,  # 3 * 3 卷积核数量
                  filter5R,  # 5 * 5 卷积核前置 1 * 1 卷积核的数量
                  filter5,  # 5 * 5 卷积核数量
                  proj,
                  name=None):
        # 1 * 1 conv
        conv1 = self.conv_layer(
            input=input,
            num_filters=filter1,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_1x1")
        # 1* 1 conv => 3 * 3 conv
        conv3r = self.conv_layer(
            input=input,
            num_filters=filter3R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3_reduce")
        conv3 = self.conv_layer(
            input=conv3r,
            num_filters=filter3,
            filter_size=3,
            stride=1,
            act=None,
            name="inception_" + name + "_3x3")
        # 1 * 1 conv => 5 * 5 conv
        conv5r = self.conv_layer(
            input=input,
            num_filters=filter5R,
            filter_size=1,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5_reduce")
        conv5 = self.conv_layer(
            input=conv5r,
            num_filters=filter5,
            filter_size=5,
            stride=1,
            act=None,
            name="inception_" + name + "_5x5")
        # 3 * 3 max pooling => 1 * 1 conv
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=3,
            pool_stride=1,
            pool_padding=1,
            pool_type='max')
        convprj = fluid.layers.conv2d(
            input=pool,
            filter_size=1,
            num_filters=proj,
            stride=1,
            padding=0,
            name="inception_" + name + "_3x3_proj",
            param_attr=ParamAttr(
                name="inception_" + name + "_3x3_proj_weights"),
            bias_attr=False)
        cat = fluid.layers.concat(input=[conv1, conv3, conv5, convprj], axis=1)
        cat = fluid.layers.relu(cat)
        return cat
    
    # 定义网络结构
    def net(self, input, class_dim=1000):
        # 7 * 7 conv => 3 * 3 max pooling
        conv = self.conv_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=None,
            name="conv1")
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)
        # 1*1 conv => 3*3 conv
        conv = self.conv_layer(
            input=pool,
            num_filters=64,
            filter_size=1,
            stride=1,
            act=None,
            name="conv2_1x1")
        conv = self.conv_layer(
            input=conv,
            num_filters=192,
            filter_size=3,
            stride=1,
            act=None,
            name="conv2_3x3")
        # 3*3 max pooling
        pool = fluid.layers.pool2d(
            input=conv, pool_size=3, pool_type='max', pool_stride=2)
        # inception => incepiton => max pooling
        ince3a = self.inception(pool, 192, 64, 96, 128, 16, 32, 32, "ince3a")
        ince3b = self.inception(ince3a, 256, 128, 128, 192, 32, 96, 64,
                                "ince3b")
        pool3 = fluid.layers.pool2d(
            input=ince3b, pool_size=3, pool_type='max', pool_stride=2)
        # inception * 5 => max pooling
        ince4a = self.inception(pool3, 480, 192, 96, 208, 16, 48, 64, "ince4a")
        ince4b = self.inception(ince4a, 512, 160, 112, 224, 24, 64, 64,
                                "ince4b")
        ince4c = self.inception(ince4b, 512, 128, 128, 256, 24, 64, 64,
                                "ince4c")
        ince4d = self.inception(ince4c, 512, 112, 144, 288, 32, 64, 64,
                                "ince4d")
        ince4e = self.inception(ince4d, 528, 256, 160, 320, 32, 128, 128,
                                "ince4e")
        # inception => inception => avg pooling
        pool4 = fluid.layers.pool2d(
            input=ince4e, pool_size=3, pool_type='max', pool_stride=2)

        ince5a = self.inception(pool4, 832, 256, 160, 320, 32, 128, 128,
                                "ince5a")
        ince5b = self.inception(ince5a, 832, 384, 192, 384, 48, 128, 128,
                                "ince5b")
        pool5 = fluid.layers.pool2d(
            input=ince5b, pool_size=7, pool_type='avg', pool_stride=7)
        # 最深层次的输出 dropout => fc(softmax)
        dropout = fluid.layers.dropout(x=pool5, dropout_prob=0.4)
        out = fluid.layers.fc(input=dropout,
                              size=class_dim,
                              act='softmax',
                              param_attr=self.xavier(1024, 1, "out"),
                              name="out",
                              bias_attr=ParamAttr(name="out_offset"))
        # 最浅层的输出
        pool_o1 = fluid.layers.pool2d(
            input=ince4a, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o1 = self.conv_layer(
            input=pool_o1,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o1")
        fc_o1 = fluid.layers.fc(input=conv_o1,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o1"),
                                name="fc_o1",
                                bias_attr=ParamAttr(name="fc_o1_offset"))
        dropout_o1 = fluid.layers.dropout(x=fc_o1, dropout_prob=0.7)
        out1 = fluid.layers.fc(input=dropout_o1,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out1"),
                               name="out1",
                               bias_attr=ParamAttr(name="out1_offset"))
        # 稍浅层的输出
        pool_o2 = fluid.layers.pool2d(
            input=ince4d, pool_size=5, pool_type='avg', pool_stride=3)
        conv_o2 = self.conv_layer(
            input=pool_o2,
            num_filters=128,
            filter_size=1,
            stride=1,
            act=None,
            name="conv_o2")
        fc_o2 = fluid.layers.fc(input=conv_o2,
                                size=1024,
                                act='relu',
                                param_attr=self.xavier(2048, 1, "fc_o2"),
                                name="fc_o2",
                                bias_attr=ParamAttr(name="fc_o2_offset"))
        dropout_o2 = fluid.layers.dropout(x=fc_o2, dropout_prob=0.7)
        out2 = fluid.layers.fc(input=dropout_o2,
                               size=class_dim,
                               act='softmax',
                               param_attr=self.xavier(1024, 1, "out2"),
                               name="out2",
                               bias_attr=ParamAttr(name="out2_offset"))

        # last fc layer is "out"
        return out, out1, out2


# In[ ]:


def init_log_config():
    """
    初始化日志相关配置
    :return:
    """
    global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 日志输出的最低等级 logging.INFO
    log_path = os.path.join(os.getcwd(), 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = os.path.join(log_path, 'train.log')
    sh = logging.StreamHandler()
    fh = logging.FileHandler(log_name, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.addHandler(fh)


def init_train_parameters():
    """
    初始化训练参数，主要是初始化图片数量，类别数
    :return:
    """
    train_file_list = os.path.join(train_parameters['data_dir'], train_parameters['train_file_list'])
    label_list = os.path.join(train_parameters['data_dir'], train_parameters['label_file'])
    index = 0
    with codecs.open(label_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        for line in lines:
            parts = line.strip().split()
            # 初始化 label_dict
            train_parameters['label_dict'][parts[1]] = int(parts[0])
            index += 1
        # 初始化 class_dim
        train_parameters['class_dim'] = index
    # 初始化 image_count
    with codecs.open(train_file_list, encoding='utf-8') as flist:
        lines = [line.strip() for line in flist]
        train_parameters['image_count'] = len(lines)


def resize_img(img, target_size):
    """
    强制缩放图片
    :param img:
    :param target_size:
    :return:
    """
    target_size = input_size
    img = img.resize((target_size[1], target_size[2]), Image.BILINEAR)
    return img


def random_crop(img, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    """
    图片随机裁剪，扣取中心区域
    """
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((train_parameters['input_size'][1], train_parameters['input_size'][2]), Image.BILINEAR)
    return img


def rotate_image(img):
    """
    图像增强，增加随机旋转角度
    """
    angle = np.random.randint(-14, 15)
    img = img.rotate(angle)
    return img


def random_brightness(img):
    """
    图像增强，亮度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['brightness_prob']:
        brightness_delta = train_parameters['image_enhance_strategy']['brightness_delta']
        delta = np.random.uniform(-brightness_delta, brightness_delta) + 1
        img = ImageEnhance.Brightness(img).enhance(delta)
    return img


def random_contrast(img):
    """
    图像增强，对比度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['contrast_prob']:
        contrast_delta = train_parameters['image_enhance_strategy']['contrast_delta']
        delta = np.random.uniform(-contrast_delta, contrast_delta) + 1
        img = ImageEnhance.Contrast(img).enhance(delta)
    return img


def random_saturation(img):
    """
    图像增强，饱和度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['saturation_prob']:
        saturation_delta = train_parameters['image_enhance_strategy']['saturation_delta']
        delta = np.random.uniform(-saturation_delta, saturation_delta) + 1
        img = ImageEnhance.Color(img).enhance(delta)
    return img


def random_hue(img):
    """
    图像增强，色度调整
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    if prob < train_parameters['image_enhance_strategy']['hue_prob']:
        hue_delta = train_parameters['image_enhance_strategy']['hue_delta']
        delta = np.random.uniform(-hue_delta, hue_delta)
        img_hsv = np.array(img.convert('HSV'))
        img_hsv[:, :, 0] = img_hsv[:, :, 0] + delta
        img = Image.fromarray(img_hsv, mode='HSV').convert('RGB')
    return img


def distort_color(img):
    """
    概率的图像增强
    :param img:
    :return:
    """
    prob = np.random.uniform(0, 1)
    # Apply different distort order
    if prob < 0.35:
        img = random_brightness(img)  # 亮度
        img = random_contrast(img)  # 对比度
        img = random_saturation(img)  # 饱和度
        img = random_hue(img)  # 色度
    elif prob < 0.7:
        img = random_brightness(img)  # 亮度
        img = random_saturation(img)  # 饱和度
        img = random_hue(img)  # 色度
        img = random_contrast(img)  # 对比度
    return img


def custom_image_reader(file_list, data_dir, mode):
    """
    自定义用户图片读取器，先初始化图片种类，数量
    :param file_list:
    :param data_dir:
    :param mode:
    :return:
    """
    with codecs.open(file_list) as flist:
        lines = [line.strip() for line in flist]

    def reader():
        np.random.shuffle(lines)  # shuffle
        for line in lines:
            if mode == 'train' or mode == 'val':
                img_path, label = line.split()
                img = Image.open(img_path)
                try:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    if train_parameters['image_enhance_strategy']['need_distort'] == True:
                        img = distort_color(img)
                    if train_parameters['image_enhance_strategy']['need_rotate'] == True:
                        img = rotate_image(img)
                    if train_parameters['image_enhance_strategy']['need_crop'] == True:
                        img = random_crop(img, train_parameters['input_size'])
                    if train_parameters['image_enhance_strategy']['need_flip'] == True:
                        mirror = int(np.random.uniform(0, 2))
                        if mirror == 1:
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    # HWC--->CHW && normalized
                    img = np.array(img).astype('float32')
                    img -= train_parameters['mean_rgb']
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img *= 0.007843                 # 像素值归一化
                    yield img, int(label)
                except Exception as e:
                    pass                            # 以防某些图片读取处理出错，加异常处理
            elif mode == 'test':
                img_path = os.path.join(data_dir, line)
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = resize_img(img, train_parameters['input_size'])
                # HWC--->CHW && normalized
                img = np.array(img).astype('float32')
                img -= train_parameters['mean_rgb']
                img = img.transpose((2, 0, 1))  # HWC to CHW
                img *= 0.007843  # 像素值归一化
                yield img

    return reader


def optimizer_momentum_setting():
    """
    阶梯型的学习率适合比较大规模的训练数据
    """
    learning_strategy = train_parameters['momentum_strategy']
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    return optimizer


def optimizer_rms_setting():
    """
    阶梯型的学习率适合比较大规模的训练数据
    """
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    learning_strategy = train_parameters['rsm_strategy']
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]

    optimizer = fluid.optimizer.RMSProp(
        learning_rate=fluid.layers.piecewise_decay(boundaries, values))

    return optimizer


def optimizer_sgd_setting():
    """
    loss下降相对较慢，但是最终效果不错，阶梯型的学习率适合比较大规模的训练数据
    """
    learning_strategy = train_parameters['sgd_strategy']
    batch_size = train_parameters["train_batch_size"]
    iters = train_parameters["image_count"] // batch_size
    lr = learning_strategy['learning_rate']

    boundaries = [i * iters for i in learning_strategy["lr_epochs"]]
    values = [i * lr for i in learning_strategy["lr_decay"]]
    learning_rate = fluid.layers.piecewise_decay(boundaries, values)
    optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
    return optimizer


def optimizer_adam_setting():
    """
    能够比较快速的降低 loss，但是相对后期乏力
    """
    learning_strategy = train_parameters['adam_strategy']
    learning_rate = learning_strategy['learning_rate']
    optimizer = fluid.optimizer.Adam(learning_rate=learning_rate)
    return optimizer


def load_params(exe, program):
    if train_parameters['continue_train'] and os.path.exists(train_parameters['save_persistable_dir']):
        logger.info('load params from retrain model')
        fluid.io.load_persistables(executor=exe,
                                   dirname=train_parameters['save_persistable_dir'],
                                   main_program=program)
    elif train_parameters['pretrained'] and os.path.exists(train_parameters['pretrained_dir']):
        logger.info('load params from pretrained model')
        def if_exist(var):
            return os.path.exists(os.path.join(train_parameters['pretrained_dir'], var.name))

        fluid.io.load_vars(exe, train_parameters['pretrained_dir'], main_program=program,
                           predicate=if_exist)


def train():
    train_prog = fluid.Program()
    train_startup = fluid.Program()
    logger.info("create prog success")
    logger.info("train config: %s", str(train_parameters))
    logger.info("build input custom reader and data feeder")
    file_list = os.path.join(train_parameters['data_dir'], "train.txt")
    mode = train_parameters['mode']
    batch_reader = paddle.batch(custom_image_reader(file_list, train_parameters['data_dir'], mode),
                                batch_size=train_parameters['train_batch_size'],
                                drop_last=False)
    batch_reader = paddle.reader.shuffle(batch_reader, train_parameters['train_batch_size'])
    place = fluid.CUDAPlace(0) if train_parameters['use_gpu'] else fluid.CPUPlace()
    # 定义输入数据的占位符
    img = fluid.layers.data(name='img', shape=train_parameters['input_size'], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)

    # 选取不同的网络
    logger.info("build newwork")
    model = GoogleNet()
    out, out1, out2 = model.net(input=img, class_dim=train_parameters['class_dim'])
    cost0 = fluid.layers.cross_entropy(input=out, label=label)
    cost1 = fluid.layers.cross_entropy(input=out1, label=label)
    cost2 = fluid.layers.cross_entropy(input=out2, label=label)
    avg_cost0 = fluid.layers.mean(x=cost0)
    avg_cost1 = fluid.layers.mean(x=cost1)
    avg_cost2 = fluid.layers.mean(x=cost2)

    avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
    acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
    # 选取不同的优化器
    optimizer = optimizer_rms_setting()
    # optimizer = optimizer_momentum_setting()
    # optimizer = optimizer_sgd_setting()
    # optimizer = optimizer_adam_setting()
    optimizer.minimize(avg_cost)
    exe = fluid.Executor(place)

    main_program = fluid.default_main_program()
    exe.run(fluid.default_startup_program())
    train_fetch_list = [avg_cost.name, acc_top1.name, out.name]
    
    load_params(exe, main_program)

    # 训练循环主体
    stop_strategy = train_parameters['early_stop']
    successive_limit = stop_strategy['successive_limit']
    sample_freq = stop_strategy['sample_frequency']
    good_acc1 = stop_strategy['good_acc1']
    successive_count = 0
    stop_train = False
    total_batch_count = 0
    for pass_id in range(train_parameters["num_epochs"]):
        logger.info("current pass: %d, start read image", pass_id)
        batch_id = 0
        for step_id, data in enumerate(batch_reader()):
            t1 = time.time()
            loss, acc1, pred_ot = exe.run(main_program,
                                          feed=feeder.feed(data),
                                          fetch_list=train_fetch_list)
            t2 = time.time()
            batch_id += 1
            total_batch_count += 1
            period = t2 - t1
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            if batch_id % 10 == 0:
                logger.info("Pass {0}, trainbatch {1}, loss {2}, acc1 {3}, time {4}".format(pass_id, batch_id, loss, acc1,
                                                                                            "%2.2f sec" % period))
            # 简单的提前停止策略，认为连续达到某个准确率就可以停止了
            if acc1 >= good_acc1:
                successive_count += 1
                logger.info("current acc1 {0} meets good {1}, successive count {2}".format(acc1, good_acc1, successive_count))
                fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program,
                                              executor=exe)
                if successive_count >= successive_limit:
                    logger.info("end training")
                    stop_train = True
                    break
            else:
                successive_count = 0

            # 通用的保存策略，减小意外停止的损失
            if total_batch_count % sample_freq == 0:
                logger.info("temp save {0} batch train result, current acc1 {1}".format(total_batch_count, acc1))
                fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
        if stop_train:
            break
    logger.info("training till last epcho, end training")
    fluid.io.save_persistables(dirname=train_parameters['save_persistable_dir'],
                                           main_program=main_program,
                                           executor=exe)
    fluid.io.save_inference_model(dirname=train_parameters['save_freeze_dir'],
                                              feeded_var_names=['img'],
                                              target_vars=[out],
                                              main_program=main_program.clone(for_test=True),
                                              executor=exe)


if __name__ == '__main__':
    init_log_config()
    init_train_parameters()
    train()


# In[ ]:




