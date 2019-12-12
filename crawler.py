#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
"""
 @Time : 2019/11/14 9:57 
 @Author : ZHANG 
 @File : crawler.py 
 @Description:
"""
"""
from urllib.request import urlopen#用于获取网页
from bs4 import BeautifulSoup#用于解析网页

html = urlopen('https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md')
bsObj = BeautifulSoup(html, 'html.parser')
t1 = bsObj.find_all('a')
for t2 in t1:
    t3 = t2.get('href')
    print(t3)
"""
import wget
models = ['RetinaNet','RPN', 'CascadeMaskRCNN', 'FasterRCNN', 'CascadeRCNN', 'HybridTaskCascade', 'MaskRCNN'] #'SSD'
for mod in models:
    with open("/dlwsdata3/public/mmdetection/"+mod+"/link.txt", "r") as ins:
        for line in ins:
            #new = line.replace('https://s3.ap-northeast-2.amazonaws.com/open-mmlab', 'https://open-mmlab.oss-cn-beijing.aliyuncs.com')
            new = line.strip()
            print (new)
            out = "/dlwsdata3/public/mmdetection/"+mod + "/"
            wget.download(new, out)
            print (out)