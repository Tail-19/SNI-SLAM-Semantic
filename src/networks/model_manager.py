# This file is a part of SNI-SLAM.

from src.networks.dinov2_seg import DINO2SEG
from src.networks.mlp import head, encoder, feature_fusion #导入head, encoder, feature_fusion，一些全连接层
import torch

#ModelManager 提供了一个接口，用于管理模型的训练和推理
class ModelManager:
    def __init__(self, cfg): 
        self.dim = cfg['model']['c_dim'] #颜色的维度
        self.hidden_dim = cfg['model']['hidden_dim'] #隐藏层的维度

        self.encoder_multires = cfg['model']['encoder']['multires'] #编码器的多分辨率

        self.pretrained_model_path = cfg['model']['cnn']['pretrained_model_path'] #预训练模型的路径
        self.n_classes = cfg['model']['cnn']['n_classes'] #类别数（如何做开放类别-CLIP）

        self.img_h = cfg['cam']['H'] #图像的高度
        self.img_w = cfg['cam']['W'] #图像的宽度

        self.crop_edge = cfg['cam']['crop_edge'] #

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #device

        self.encoder = self.get_encoder().cuda() #编码器，放到cuda上
        self.head = self.get_head().cuda() #head，放到cuda上，用于获取semantic
        self.cnn = self.get_dinov2().cuda() #dinov2，放到cuda上

        self.feat_fusion = self.get_fusion().cuda() #特征融合，放到cuda上

    def train(self):
        self.feat_fusion.train() #训练特征融合
        self.encoder.train() #训练编码器
        self.head.train() #训练head

    def get_encoder(self):
        return encoder(hidden_dim=self.hidden_dim, out_dim=self.dim, multires=self.encoder_multires)

    def get_head(self):
        return head(in_dim=self.dim, hidden_dim=self.hidden_dim)

    def get_dinov2(self):
        model = DINO2SEG(img_h=self.img_h, img_w=self.img_w, num_cls=self.n_classes, edge=self.crop_edge, dim=self.dim)
        model.load_state_dict(torch.load(self.pretrained_model_path, map_location=self.device))
        return model

    def set_mode_feature(self):
        self.cnn.mode = 'mapping' #设置dinov2的模式为mapping

    def set_mode_result(self):
        self.cnn.mode = 'train' #设置dinov2的模式为train

    def get_fusion(self):
        return feature_fusion(dim=self.dim, hidden_dim=self.hidden_dim) #返回特征融合

    def get_share_memory(self):
        # share memory
        self.feat_fusion.share_memory() 
        self.encoder.share_memory()
        self.head.share_memory()
        self.cnn.share_memory()

        return self