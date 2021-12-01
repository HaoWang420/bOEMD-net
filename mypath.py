class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'QUBIQ':
            return r'/data/ssd/wanghao/datasets/QUBIQ'
        if dataset == 'uncertain-brats':
            return r''
        if dataset == 'brats':
            return '/home/cvip/data/wanghao/datasets/braTS_jpg'
        if dataset == 'lidc':
            return '/data/ssd/wanghao/datasets/'
        if dataset == 'processed-lidc':
            return "/data/ssd/datasets/processed_LIDC/"
        else:
            raise NotImplementedError
