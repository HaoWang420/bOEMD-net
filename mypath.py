class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'QUBIQ':
            return r'/home/qingqiao/bAttenUnet_test/qubiq'
        if dataset == 'uncertain-brate':
            return r'/home/qingqiao/bAttenUnet_test/qubiq'
        if dataset == 'brats':
            return '/home/cvip/data/wanghao/datasets/braTS_jpg'
        else:
            raise NotImplementedError
