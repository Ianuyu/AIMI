import numpy as np

# def zscore_per_channel_global(train, test, eps=1e-6):
#         mean = train.mean(axis=(0,1,3), keepdims=True)      # (1,1,C,1)
#         std  = train.std (axis=(0,1,3), keepdims=True) + eps
#         return (train - mean) / std, (test - mean) / std

def read_bci_data():
    S4b_train  = np.load('S4b_train.npz',  allow_pickle=False, mmap_mode='r')
    X11b_train = np.load('X11b_train.npz', allow_pickle=False, mmap_mode='r')
    S4b_test   = np.load('S4b_test.npz',   allow_pickle=False, mmap_mode='r')
    X11b_test  = np.load('X11b_test.npz',  allow_pickle=False, mmap_mode='r')

    train_data  = np.concatenate([S4b_train['signal'],  X11b_train['signal']],  axis=0)
    train_label = np.concatenate([S4b_train['label'],   X11b_train['label']],   axis=0)
    test_data   = np.concatenate([S4b_test['signal'],   X11b_test['signal']],   axis=0)
    test_label  = np.concatenate([S4b_test['label'],    X11b_test['label']],    axis=0)

    train_label = (train_label - 1).astype(np.int64, copy=False)
    test_label  = (test_label  - 1).astype(np.int64,  copy=False)

    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2)).astype(np.float32, copy=False)
    test_data  = np.transpose(np.expand_dims(test_data,  axis=1), (0, 1, 3, 2)).astype(np.float32, copy=False)

    if np.isnan(train_data).any():
        train_mean = np.nanmean(train_data).astype(np.float32)
        train_data = np.nan_to_num(train_data, nan=train_mean)
    if np.isnan(test_data).any():
        test_mean = np.nanmean(test_data).astype(np.float32)
        test_data = np.nan_to_num(test_data, nan=test_mean)

    # train_data, test_data = zscore_per_channel_global(train_data, test_data)
    return train_data, train_label, test_data, test_label
