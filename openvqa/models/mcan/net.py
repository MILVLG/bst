
from openvqa.utils.make_mask import make_mask
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Img_Adapter, Lang_Adapter
from openvqa.ops.slimmable_ops import SlimmableAttFlat, SlimmableLayerNorm, SlimmableLinear

import torch.nn as nn
import torch

# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.lang_adapter = Lang_Adapter(__C)
        self.img_adapter = Img_Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = SlimmableAttFlat(__C)
        self.attflat_lang = SlimmableAttFlat(__C)

        # Classification layers
        self.proj_norm = SlimmableLayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = SlimmableLinear(__C.FLAT_OUT_SIZE, answer_size, style='end')

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix, width=1, depth=1):

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)
        lang_feat = self.lang_adapter(lang_feat, width)

        img_feat, img_feat_mask = self.img_adapter(frcn_feat, grid_feat, bbox_feat, width)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask,
            width,
            depth
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask,
            width
        )

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask,
            width
        )

        # Classification layers
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat, width)
        proj_feat = self.proj(proj_feat, width)

        return proj_feat
