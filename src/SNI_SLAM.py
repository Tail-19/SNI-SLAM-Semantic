# This file is a part of SNI-SLAM

import os
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.multiprocessing as mp

from src import config
from src.Mapper import Mapper
from src.Tracker import Tracker
from src.utils.datasets import get_dataset
from src.utils.Logger import Logger
from src.utils.Mesher import Mesher
from src.utils.Renderer import Renderer

from src.networks.model_manager import ModelManager

torch.multiprocessing.set_sharing_strategy('file_system')

import wandb


class SNI_SLAM():
    # 这是SNI_SLAM的主类，用于管理整个SLAM流程
    def __init__(self, cfg, args):
        # 初始化系统，包括读取配置、创建输出目录等
        self.cfg = cfg #传入config文件的键值对
        self.args = args #传入命令行参数

        self.verbose = cfg['verbose'] #是否输出详细信息
        self.device = cfg['device'] #使用的设备
        self.dataset = cfg['dataset']   #数据集名称
        self.truncation = cfg['model']['truncation'] #截断值
        
        self.semantic_only = cfg['semantic_only'] #是否只输出语义信息

        #如果没有在命令行指定输出目录，则使用配置文件中的输出目录
        if args.output is None: 
            self.output = cfg['data']['output'] 
        else:
            self.output = args.output 
        self.ckptsdir = os.path.join(self.output, 'ckpts') #模型检查点目录，放在输出目录下的ckpts文件夹中
        os.makedirs(self.output, exist_ok=True) #创建输出目录 
        os.makedirs(self.ckptsdir, exist_ok=True) #exist_ok=True表示如果目录已经存在则不会报错
        os.makedirs(f'{self.output}/mesh', exist_ok=True) #创建mesh文件夹

        #读取相机内参
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = cfg['cam']['H'], cfg['cam'][
            'W'], cfg['cam']['fx'], cfg['cam']['fy'], cfg['cam']['cx'], cfg['cam']['cy']
        self.update_cam() #根据CROP_SIZE和CROP_EDGE更新相机内参

        #读取模型配置，这里的model作为共享的解码器
        model = config.get_model(cfg)
        self.shared_decoders = model #共享的解码器

        self.scale = cfg['scale'] #读取缩放比例

        self.load_bound(cfg) #加载场景边界
        self.init_planes(cfg) #初始化平面张量

        self.enable_wandb = cfg['func']['enable_wandb'] #是否启用wandb
        if self.enable_wandb: #wanb是一个在线的实验记录工具，可以用于记录实验的参数、结果等
            self.wandb_run = wandb.init(project="sni_slam") #初始化wandb

        # need to use spawn，spawn是一种多进程的方式，可以避免一些多线程的问题
        try:
            mp.set_start_method('spawn', force=True) #设置多进程的启动方式，mp是torch.multiprocessing，这里设置为spawn
        except RuntimeError:
            pass

        self.frame_reader = get_dataset(cfg, args, self.scale) #读取数据集，这里frame_reader是一个迭代器
        self.n_img = len(self.frame_reader) #获取数据集的总帧数
        
        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4), device=self.device) #估计的相机位姿，n_img*4*4的张量，4*4是相机位姿矩阵
        self.estimate_c2w_list.share_memory_() #共享内存，多进程共享数据，share_memory_()是pytorch的方法，用于在多进程中共享数据
        self.gt_c2w_list = (torch.zeros((self.n_img, 4, 4))) #ground truth真实的相机位姿，n_img*4*4的张量
        self.gt_c2w_list.share_memory_() #共享内存
        
        self.idx = torch.zeros((1)).int() #当前帧的索引，用于记录当前处理的帧
        self.idx.share_memory_() #共享内存
        self.mapping_first_frame = torch.zeros((1)).int() #是否是第一帧，用于mapping线程的同步
        self.mapping_first_frame.share_memory_() #共享内存

        # the id of the newest frame Mapper is processing
        self.mapping_idx = torch.zeros((1)).int() #mapping线程正在处理的帧的索引
        self.mapping_idx.share_memory_()
        self.mapping_cnt = torch.zeros((1)).int()  # mapping线程的计数器，理论上和mapping_idx，idx是一样的
        self.mapping_cnt.share_memory_()

        ## Moving feature planes and decoders to the processing device
        if not self.semantic_only:
            for shared_planes in [self.shared_planes_xy, self.shared_planes_xz, self.shared_planes_yz]:
                for i, plane in enumerate(shared_planes):
                    plane = plane.to(self.device)
                    plane.share_memory_()
                    shared_planes[i] = plane

            for shared_c_planes in [self.shared_c_planes_xy, self.shared_c_planes_xz, self.shared_c_planes_yz]:
                for i, plane in enumerate(shared_c_planes):
                    plane = plane.to(self.device)
                    plane.share_memory_()
                    shared_c_planes[i] = plane

        for shared_s_planes in [self.shared_s_planes_xy, self.shared_s_planes_xz, self.shared_s_planes_yz]:
            for i, plane in enumerate(shared_s_planes):
                plane = plane.to(self.device)
                plane.share_memory_()
                shared_s_planes[i] = plane

        # 把解码器和模型管理器移动到处理设备上，然后共享内存
        self.shared_decoders = self.shared_decoders.to(self.device)
        self.shared_decoders.share_memory()
        self.model_manager = ModelManager(cfg)
        self.model_manager.get_share_memory()

        # 初始化组件
        self.renderer = Renderer(cfg, self)
        self.mesher = Mesher(cfg, args, self)
        self.logger = Logger(self)
        self.mapper = Mapper(cfg, args, self)
        self.tracker = Tracker(cfg, args, self) 
        self.print_output_desc()

    def print_output_desc(self):
        # 打印输出信息，说明结果文件的存储位置
        print(f"INFO: The output folder is {self.output}")
        print(
            f"INFO: The GT, generated and residual depth/color images can be found under " +
            f"{self.output}/tracking_vis/ and {self.output}/mapping_vis/")
        print(f"INFO: The mesh can be found under {self.output}/mesh/")
        print(f"INFO: The checkpoint can be found under {self.output}/ckpt/")

    def update_cam(self):
        # 根据预处理配置更新相机内参
        """
        Update the camera intrinsics according to pre-processing config, 
        such as resize or edge crop.
        """
        # resize the input images to crop_size (variable name used in lietorch)
        if 'crop_size' in self.cfg['cam']: #如果配置文件中有crop_size字段
            crop_size = self.cfg['cam']['crop_size'] #读取crop_size字段
            sx = crop_size[1] / self.W #CROPSIZE/相机宽度
            sy = crop_size[0] / self.H #计算缩放比例
            #新内参=CROPSIZE/相机size*原内参（将相机的参数缩放到CROP之后的大小）
            self.fx = sx*self.fx 
            self.fy = sy*self.fy 
            self.cx = sx*self.cx 
            self.cy = sy*self.cy
            #更新相机的宽高为CROP之后的大小
            self.W = crop_size[1]
            self.H = crop_size[0]

        # croping will change H, W, cx, cy, so need to change here
        #Crop_edge是在crop_size基础上再裁剪的边缘大小，如果有的话，需要更新相机内参
        if self.cfg['cam']['crop_edge'] > 0:
            #对于宽和高而言，减去crop_edge*2
            self.H -= self.cfg['cam']['crop_edge']*2 
            self.W -= self.cfg['cam']['crop_edge']*2
            #cx和cy是相对于左上角的，所以需要减去crop_edge
            self.cx -= self.cfg['cam']['crop_edge']
            self.cy -= self.cfg['cam']['crop_edge']

    def load_bound(self, cfg):
        # 加载并缩放场景边界用于后续处理
        """
        Pass the scene bound parameters to different decoders and self.

        Args:
            cfg (dict): parsed config dict.
        """

        # scale the bound if there is a global scaling factor
        #np.array(cfg['mapping']['bound'])是一个列表，表示场景的边界,长度为6,因为是3D场景，所以有6个值
        self.bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*self.scale).float() #读取场景边界，乘以缩放因子，这个bound来源于配置文件，是一个列表，表示场景的边界
        bound_dividable = cfg['planes_res']['bound_dividable'] #读取bound_dividable字段，用于扩大边界，使其可以被planes_res整除
        # enlarge the bound a bit to allow it dividable by bound_dividable
        self.bound[:, 1] = (((self.bound[:, 1]-self.bound[:, 0]) /
                            bound_dividable).int()+1)*bound_dividable+self.bound[:, 0] #扩大边界，使其可以被bound_dividable整除
        self.shared_decoders.bound = self.bound #将边界传递给解码器

    def init_planes(self, cfg):
        # 初始化平面张量，用于三维环境的特征表示
        """
        Initialize the feature planes.

        Args:
            cfg (dict): parsed config dict.
        """
        if not self.semantic_only:
            #depth planes
            self.coarse_planes_res = cfg['planes_res']['coarse']
            self.fine_planes_res = cfg['planes_res']['fine']
            #color planes
            self.coarse_c_planes_res = cfg['c_planes_res']['coarse']
            self.fine_c_planes_res = cfg['c_planes_res']['fine']
        #semantic planes
        self.coarse_s_planes_res = cfg['s_planes_res']['coarse']
        self.fine_s_planes_res = cfg['s_planes_res']['fine']

        c_dim = cfg['model']['c_dim'] #读取c_dim字段，表示颜色通道数
        # print("c_dim", c_dim)
        xyz_len = self.bound[:, 1]-self.bound[:, 0] #计算边界的长度，max-min

        ####### Initializing Planes ############
        if not self.semantic_only:
            planes_xy, planes_xz, planes_yz = [], [], [] #三个方向的平面
            c_planes_xy, c_planes_xz, c_planes_yz = [], [], [] #三个方向的颜色平面
        s_planes_xy, s_planes_xz, s_planes_yz = [], [], []  # 三个方向的语义平面
        ####### Plane Resolution ############
        if not self.semantic_only:
            planes_res = [self.coarse_planes_res, self.fine_planes_res]
            c_planes_res = [self.coarse_c_planes_res, self.fine_c_planes_res]
        s_planes_res = [self.coarse_s_planes_res, self.fine_s_planes_res]

        planes_dim = c_dim #平面的维度等于颜色通道数
        if not self.semantic_only:
            for grid_res in planes_res:
                grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
                grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
                planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
                planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
                planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

            for grid_res in c_planes_res:
                grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
                grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
                #这里append的是一个正态分布的张量，均值为0，标准差为0.01
                #它是对颜色平面权重的一种随机初始化，避免所有平面权重都从相同值开始，从而在训练过程中有更丰富的学习空间并能更好地收敛。
                c_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01)) #创建一个正态分布的张量,均值为0，标准差为0.01,用于表示颜色平面x-y平面
                c_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01)) #创建一个正态分布的张量,均值为0，标准差为0.01,用于表示颜色平面x-z平面
                c_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        for grid_res in s_planes_res:
            grid_shape = list(map(int, (xyz_len / grid_res).tolist()))
            grid_shape[0], grid_shape[2] = grid_shape[2], grid_shape[0]
            s_planes_xy.append(torch.empty([1, planes_dim, *grid_shape[1:]]).normal_(mean=0, std=0.01))
            s_planes_xz.append(torch.empty([1, planes_dim, grid_shape[0], grid_shape[2]]).normal_(mean=0, std=0.01))
            s_planes_yz.append(torch.empty([1, planes_dim, *grid_shape[:2]]).normal_(mean=0, std=0.01))

        if not self.semantic_only:
            self.shared_planes_xy = planes_xy
            self.shared_planes_xz = planes_xz
            self.shared_planes_yz = planes_yz

            self.shared_c_planes_xy = c_planes_xy 
            self.shared_c_planes_xz = c_planes_xz
            self.shared_c_planes_yz = c_planes_yz

        self.shared_s_planes_xy = s_planes_xy
        self.shared_s_planes_xz = s_planes_xz
        self.shared_s_planes_yz = s_planes_yz

    def tracking(self, rank):
        # 跟踪线程，等待首帧mapping完成后开始
        """
        Tracking Thread.

        Args:
            rank (int): Thread ID.
        """

        # should wait until the mapping of first frame is finished
        while True:
            if self.mapping_first_frame[0] == 1: #如果mapping_first_frame为1，表示mapping线程已经处理完第一帧
                break #跳出循环，开始跟踪
            time.sleep(1)

        self.tracker.run() #开始跟踪

    def mapping(self, rank):
        # 建图线程，负责运行Mapper组件
        """
        Mapping Thread.

        Args:
            rank (int): Thread ID.
        """

        self.mapper.run()

    def run(self):
        # 启动并管理跟踪与建图进程
        """
        Dispatch Threads.
        """

        processes = [] #进程列表
        for rank in range(0, 2): #两个进程，一个跟踪，一个建图
            if rank == 0: #跟踪线程
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1: #建图线程
                p = mp.Process(target=self.mapping, args=(rank, ))

            p.start()
            processes.append(p) #将进程加入进程列表
        for p in processes:
            p.join()

# This part is required by torch.multiprocessing
if __name__ == '__main__':
    pass
