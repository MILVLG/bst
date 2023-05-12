
def feat_filter(dataset, frcn_feat, grid_feat, bbox_feat):
    feat_dict = {}

    if dataset in ['vqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['BBOX_FEAT'] = bbox_feat

    elif dataset in ['gqa']:
        feat_dict['FRCN_FEAT'] = frcn_feat
        feat_dict['BBOX_FEAT'] = bbox_feat

    else:
        exit(-1)

    return feat_dict


