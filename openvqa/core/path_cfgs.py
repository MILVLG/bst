
import os

class PATH:
    def __init__(self):
        self.init_path()


    def init_path(self):

        self.DATA_ROOT = './data'

        self.DATA_PATH = {
            'vqa': self.DATA_ROOT + '/vqa',
            'gqa': self.DATA_ROOT + '/gqa',
        }


        self.FEATS_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/feats' + '/train2014',
                'val': self.DATA_PATH['vqa'] + '/feats' + '/val2014',
                'test': self.DATA_PATH['vqa'] + '/feats' + '/test2015',
            },
            'gqa': {
                'default-frcn': self.DATA_PATH['gqa'] + '/feats' + '/gqa-frcn',
            },
        }


        self.RAW_PATH = {
            'vqa': {
                'train': self.DATA_PATH['vqa'] + '/raw' + '/train_trainval_questions.json',
                'train-anno': self.DATA_PATH['vqa'] + '/raw' + '/train_trainval_annotations.json',
                'val': self.DATA_PATH['vqa'] + '/raw' + '/devval_questions.json',
                'val-anno': self.DATA_PATH['vqa'] + '/raw' + '/devval_annotations.json',
                'vg': self.DATA_PATH['vqa'] + '/raw' + '/vg_questions.json',
                'vg-anno': self.DATA_PATH['vqa'] + '/raw' + '/vg_annotations.json',
                'test': self.DATA_PATH['vqa'] + '/raw' + '/test_questions.json',
            },
            'gqa': {
                'train': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/train_balanced_questions.json',
                'val': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_balanced_questions.json',
                'testdev': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_balanced_questions.json',
                'test': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/submission_all_questions.json',
                'val_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/val_all_questions.json',
                'testdev_all': self.DATA_PATH['gqa'] + '/raw' + '/questions1.2/testdev_all_questions.json',
                'train_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/train_choices',
                'val_choices': self.DATA_PATH['gqa'] + '/raw' + '/eval/val_choices.json',
            },
        }


        self.SPLITS = {
            'vqa': {
                'train': '',
                'val': 'val',
                'test': 'test',
            },
            'gqa': {
                'train': '',
                'val': 'testdev',
                'test': 'test',
            },
        }


        self.RESULT_PATH = './results/result_test'
        self.PRED_PATH = './results/pred'
        self.CACHE_PATH = './results/cache'
        self.LOG_PATH = './results/log'
        self.CKPTS_PATH = './ckpts'

        if 'result_test' not in os.listdir('./results'):
            os.mkdir('./results/result_test')

        if 'pred' not in os.listdir('./results'):
            os.mkdir('./results/pred')

        if 'cache' not in os.listdir('./results'):
            os.mkdir('./results/cache')

        if 'log' not in os.listdir('./results'):
            os.mkdir('./results/log')

        if 'ckpts' not in os.listdir('./'):
            os.mkdir('./ckpts')


    def check_path(self, dataset=None):
        print('Checking dataset ........')


        if dataset:
            for item in self.FEATS_PATH[dataset]:
                if not os.path.exists(self.FEATS_PATH[dataset][item]):
                    print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

            for item in self.RAW_PATH[dataset]:
                if not os.path.exists(self.RAW_PATH[dataset][item]):
                    print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                    exit(-1)

        else:
            for dataset in self.FEATS_PATH:
                for item in self.FEATS_PATH[dataset]:
                    if not os.path.exists(self.FEATS_PATH[dataset][item]):
                        print(self.FEATS_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

            for dataset in self.RAW_PATH:
                for item in self.RAW_PATH[dataset]:
                    if not os.path.exists(self.RAW_PATH[dataset][item]):
                        print(self.RAW_PATH[dataset][item], 'NOT EXIST')
                        exit(-1)

        print('Finished!')
        print('')

