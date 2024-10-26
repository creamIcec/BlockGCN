from .feeder_xyz import Feeder;
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from . import tools;

coco_pairs = [(1, 6), (2, 1), (3, 1), (4, 2), (5, 3), (6, 7), (7, 1), (8, 6), (9, 7), (10, 8), (11, 9),
                (12, 6), (13, 7), (14, 12), (15, 13), (16, 14), (17, 15)]
class FeederUAVHuman(Feeder):
    #构造函数初始化
    def __init__(self, 
                 data_path, 
                 label_path=None, 
                 p_interval=1, 
                 data_split='train', 
                 random_choose=False, 
                 random_shift=False,
                 random_move=False, 
                 random_rot=False, 
                 window_size=-1, 
                 normalization=False, 
                 debug=False, 
                 use_mmap=False,
                 bone=False, 
                 vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data_split = data_split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

   # 加载数据的方法
    def load_data(self):

        # 两个都加载, 按照不同阶段返回东西就好
        # data和label可以代表训练集的，也可以代表测试集的
        data = np.load(self.data_path);
        label = np.load(self.label_path);

        #TODO 重采样

        # 这里sample_name需要一点判断，还是写上好了
        data_type_name = 'test_' if self.data_split == 'test' else 'train_';

        # 刚刚大概看了一下，还是按照一起加载的逻辑来写比较好(先暂时这样), 不然其他地方也要改
        if not self.debug:
            self.data = data;
            self.label = label;
            self.sample_name = [data_type_name + str(i) for i in range(len(self.data))];  #还是给一个sample_name吧
        else:
            self.data = data[0:100];
            self.label = label[0:100];
            self.sample_name = [data_type_name + str(i) for i in range(100)];

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)       
        if(valid_frame_num == 0):
            return torch.from_numpy(np.zeros((3, 64, 17, 2))), torch.from_numpy(np.zeros((3, 64, 17, 2))), label, index;
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        center_index = 8;
        joint = data_numpy
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            joint = data_numpy
        if self.bone:
            from .bone_pairs import coco_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in coco_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

        # keep spine center's trajectory !!! modified on July 4th, 2022
            bone_data_numpy[:, :, center_index] = data_numpy[:, :, center_index]
            data_numpy = bone_data_numpy

        # for joint modality
        # separate trajectory from relative coordinate to each frame's spine center
        else:
            # # there's a freedom to choose the direction of local coordinate axes!
            trajectory = data_numpy[:, :, center_index]
            # let spine of each frame be the joint coordinate center
            data_numpy = data_numpy - data_numpy[:, :, center_index:center_index+1]

            data_numpy[:, :, center_index] = trajectory

        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]

            data_numpy[:, -1] = 0

        return joint, data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod