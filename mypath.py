class Path(object):
    @staticmethod
    def getPath(dataset):
        if dataset == 'QUBIQ':
            return '/path/to/QUBIQ'
        if dataset == 'lidc':
            return '/path/to/lidc/'
        if dataset == 'lidc-patient':
            return "/path/to/processed_LIDC/"
        else:
            raise NotImplementedError
