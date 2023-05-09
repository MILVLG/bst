
import torch.nn as nn
import torch
from openvqa.core.base_dataset import BaseAdapter
from openvqa.utils.make_mask import make_mask
from openvqa.ops.slimmable_ops import SlimmableLinear, SlimmableLayerNorm


class Img_Adapter(BaseAdapter):
    def __init__(self, __C):
        super(Img_Adapter, self).__init__(__C)
        self.__C = __C

    def bbox_proc(self, bbox):
        area = (bbox[:, :, 2] - bbox[:, :, 0]) * (bbox[:, :, 3] - bbox[:, :, 1])
        return torch.cat((bbox, area.unsqueeze(2)), -1)

    def vqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = SlimmableLinear(imgfeat_linear_size, __C.HIDDEN_SIZE, style="start")
        self.frcn_ln = SlimmableLayerNorm(__C.HIDDEN_SIZE)


    def gqa_init(self, __C):
        imgfeat_linear_size = __C.FEAT_SIZE['gqa']['FRCN_FEAT_SIZE'][1]
        if __C.USE_BBOX_FEAT:
            self.bbox_linear = nn.Linear(5, __C.BBOXFEAT_EMB_SIZE)
            imgfeat_linear_size += __C.BBOXFEAT_EMB_SIZE
        self.frcn_linear = SlimmableLinear(imgfeat_linear_size, __C.HIDDEN_SIZE, style="start")


    def vqa_forward(self, feat_dict, width):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat, width)
        img_feat = self.frcn_ln(img_feat, width)

        return img_feat, img_feat_mask


    def gqa_forward(self, feat_dict, width):
        frcn_feat = feat_dict['FRCN_FEAT']
        bbox_feat = feat_dict['BBOX_FEAT']
        # grid_feat = feat_dict['GRID_FEAT']

        img_feat_mask = make_mask(frcn_feat)

        if self.__C.USE_BBOX_FEAT:
            bbox_feat = self.bbox_proc(bbox_feat)
            bbox_feat = self.bbox_linear(bbox_feat)
            frcn_feat = torch.cat((frcn_feat, bbox_feat), dim=-1)
        img_feat = self.frcn_linear(frcn_feat, width)

        return img_feat, img_feat_mask


class Lang_Adapter(nn.Module):
    def __init__(self, __C):
        super(Lang_Adapter, self).__init__()
        self.__C = __C
        self.lang_linear = SlimmableLinear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE, style="start")
        self.lang_ln = SlimmableLayerNorm(__C.HIDDEN_SIZE)

    def forward(self, lang_feat, width):
        lang_feat = self.lang_linear(lang_feat, width)
        lang_feat = self.lang_ln(lang_feat, width)

        return lang_feat



