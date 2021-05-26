import numpy as np


if __name__ == '__main__':
    train = np.load('train.npz')
    np.savez('train_1kk.npz', inputs=train['inputs'][:1_000_000], targets=train['targets'][:1_000_000])
    del train
    print('train saved')

    test = np.load('test.npz')
    np.savez('test_1kk.npz', inputs=test['inputs'][:50_000], targets=test['targets'][:50_000])
    del test
    print('test saved')




