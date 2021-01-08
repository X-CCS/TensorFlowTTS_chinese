# -*- coding:utf-8 -*-
import os

class config:

    CWD = os.getcwd()
    SUFFIX = '' if 'pinyin' in CWD else './'
    MY_DICT = '/home/project/TensorFlowTTS_chinese/data/mydict.dic'
    WORDS_MAPPING_PATH = "/home/project/TensorFlowTTS_chinese/data/words_mapping.csv"
    LOG_PATH = '/home/project/TensorFlowTTS_chinese/data/logs/server_{}.log'
    PINYIN_TXT = "/home/project/TensorFlowTTS_chinese/data/pinyin.txt"
    LARGE_PINYIN_TXT = "/home/project/TensorFlowTTS_chinese/data/large_pinyin.txt"

    ch_regex = "[\u4e00-\u9fa5]"
    amap1 = ['', '万', '亿']
    amap2 = ['', '十', '百', '千']
    digit = {'0':'零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

    tone = {'#2': 0, '#1': 0, '#4': 0, '#3': 0} # '#1', '#2', '#3', '#4'
    toneMap = { "ā": "a1", "á": "a2", "ǎ": "a3", "à": "a4", "ō": "o1", "ó": "o2", "ǒ": "o3", "ò": "o4", "ē": "e1", "é": "e2", "ě": "e3", "è": "e4",
		    "ī": "i1", "í": "i2", "ǐ": "i3", "ì": "i4", "ū": "u1", "ú": "u2", "ǔ": "u3", "ù": "u4", "ü": "v0", "ǖ": "v1", "ǘ": "v2", "ǚ": "v3",
		    "ǜ": "v4", "ń": "n2", "ň": "n3", "": "m2" }

    tacotron2_baker = "/home/project/TensorFlowTTS_chinese/data/tacotron2.baker.v1.yaml"
    tacotron2_pretrained_path = "/home/project/TensorFlowTTS_chinese/data/model/model-100000.h5"

    multiband_melgan_baker = "/home/project/TensorFlowTTS_chinese/data/multiband_melgan.baker.v1.yaml"
    multiband_melgan_pretrained_path = "/home/project/TensorFlowTTS_chinese/data/model/generator-920000.h5"
    
    baker_mapper_pretrained_path = "/home/project/TensorFlowTTS_chinese/data/baker_mapper.json"

