class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'QUBIQ':
            return r'/data/sdb/wanghao/datasets/QUBIQ'
        if dataset == 'uncertain-brats':
            return r''
        if dataset == 'brats':
            return '/home/cvip/data/wanghao/datasets/braTS_jpg'
        if dataset == 'lidc':
            return '/data/sdb/wanghao/datasets/'
        if dataset == 'lidc-patient':
            return "/data/sdb/datasets/processed_LIDC/"
        if dataset == "liver":
            return "/data/sdb/qingqiao/workspaces/datasets/liver_datasets/processed"
        else:
            raise NotImplementedError
