class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'QUBIQ':
            return r'/home/qingqiao/bAttenUnet_test/qubiq'
        if dataset == 'uncertain-brats':
            return r'/home/qingqiao/bAttenUnet_test/qubiq'
        if dataset == 'brats':
            return '/home/cvip/data/wanghao/datasets/braTS_jpg'
        if dataset == 'lidc':
            return '/data/ssd/wanghao/datasets/'
        else:
            raise NotImplementedError
