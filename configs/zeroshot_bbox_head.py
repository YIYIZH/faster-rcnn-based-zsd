#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/11/26 17:09 
 @Author : ZHANG 
 @File : zeroshot_bbox_head.py 
 @Description:
"""
import torch.nn as nn

from mmdet.models.registry import HEADS
from mmdet.models.utils import ConvModule
from mmdet.models.bbox_heads import ConvFCBBoxHead

@HEADS.register_module
class ZeroshotBBoxHead(ConvFCBBoxHead):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(ZeroshotBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)