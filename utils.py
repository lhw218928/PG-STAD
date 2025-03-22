import os
import torch
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader,Dataset,SubsetRandomSampler

def get_data(dataset, max_train_size=None, max_test_size=None,
             normalize=False, train_start=0, test_start=0):
    """
    Get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
    """
    prefix = "datasets"
    if str(dataset).startswith("machine"):
        prefix += "/ServerMachineDataset/processed"
    elif dataset in ["MSL", "SMAP","SWAT"]:
        prefix += "/data/processed"
    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)
    print("train: ", train_start, train_end)
    print("test: ", test_start, test_end)
    x_dim = get_data_dim(dataset)
    f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
    if dataset == "SWAT":
        train_data = np.load(f).reshape(-1, x_dim)[train_start:train_end, :]
    else:
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
    # print(type(train_data))
    f.close()
    try:
        f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
        if dataset == "SWAT":
            test_data = np.load(f).reshape(-1, x_dim)[test_start:test_end, :]
        else:
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
        f.close()
    except (KeyError, FileNotFoundError):
        test_data = None
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        if dataset == "SWAT":
            test_label = np.load(f).reshape(-1)[test_start:test_end]
        else:
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
        f.close()
    except (KeyError, FileNotFoundError):
        test_label = None

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    print("train set shape: ", train_data.shape)
    print("test set shape: ", test_data.shape)
    print("test set label shape: ", None if test_label is None else test_label.shape)
    return (train_data, None), (test_data, test_label)

def get_data_dim(dataset):
    """
    :param dataset: Name of dataset
    :return: Number of dimensions in data
    """
    if dataset == "SMAP":
        return 25
    elif dataset == "MSL":
        return 55
    elif dataset == "SWAT":
        return 51
    elif str(dataset).startswith("machine"):
        return 38
    else:
        # 后续添加WADI
        raise ValueError("unknown dataset " + str(dataset))


class SlidingWindowDataset(Dataset):
    def __init__(self, data, window_size, window_num, target_dim=None):
        """
        初始化滑动窗口数据集。

        参数：
            data (np.array or torch.Tensor): 时间序列数据，形状为 (T, N)，T是时间点数量，N是特征数量。
            window_size (int): 每个窗口的大小。
            window_num (int): 输入x包含的窗口数量。
            target_dim (int or list): 目标变量的维度索引，默认为None（使用所有维度）。
        """
        self.data = data
        self.window_size = window_size
        self.window_num = window_num
        self.target_dim = target_dim
        self.stride = window_size

    def __len__(self):
        """
        返回数据集中样本的数量。
        """
        # 样本数量 = (总时间点数量 - (window_num + 1) * window_size) // window_size + 1
        return (len(self.data) - (self.window_num + 1) * self.window_size) // self.window_size + 1

    def __getitem__(self, index):
        """
        根据索引返回一个样本。

        参数：
            index (int): 样本索引。

        返回：
            x (torch.Tensor): 输入数据，形状为 (window_num * window_size, N)。
            y (torch.Tensor): 输出数据，形状为 (window_size, N) 或 (window_size, target_dim)。
        """
        start = index * self.stride
        end = start + self.window_num * self.window_size
        y_start = end
        y_end = y_start + self.window_size

        # 获取输入x和输出y
        x = self.data[start:end, :]  # 形状为 (window_num * window_size, N)
        y = self.data[y_start:y_end, :]  # 形状为 (window_size, N)

        # # 如果指定了目标维度，只取目标维度
        # if self.target_dim is not None:
        #     y = y[:, :self.target_dim]  # 形状为 (window_size, target_dim)

        return x, y


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader



def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)
    print("Data normalized")

    return data, scaler

def get_target_dims(dataset):
    """
    :param dataset: Name of dataset
    :return: index of data dimension that should be modeled (forecasted and reconstructed),
                     returns None if all input dimensions should be modeled
    """
    if dataset == "SMAP" or dataset =="MSL":
        return 1
    elif dataset == "SWAT":
        return 25
    elif dataset == "SMD":
        return None
    else:
        raise ValueError("unknown dataset " + str(dataset))

# def batch_granger_causality(data, max_lag=4):
#     """
#     对输入数据进行批量格兰杰因果关系矩阵分析。
#
#     参数：
#         data (np.array): 输入数据，形状为 [b, w, N]，其中：
#             - b 是批次大小，
#             - w 是时间序列窗口大小，
#             - N 是特征数量。
#         max_lag (int): 格兰杰因果检验的最大滞后阶数，默认为 4。
#
#     返回：
#         causality_matrices (np.array): 因果关系矩阵，形状为 [b, N, N]。
#     """
#     b, w, N = data.shape
#     causality_matrices = np.zeros((b, N, N))  # 初始化因果关系矩阵
#
#     for batch_idx in range(b):  # 遍历每个批次
#         batch_data = data[batch_idx, :, :]  # 获取当前批次的数据 [w, N]
#         df = pd.DataFrame(batch_data, columns=[f'X{i+1}' for i in range(N)])  # 转换为DataFrame
#
#         for i in range(N):  # 遍历每个特征
#             for j in range(N):
#                 if i != j:  # 排除自身
#                     # 提取两个变量的时间序列
#                     test_data = df[[df.columns[j], df.columns[i]]]  # X_j -> X_i
#                     # 进行格兰杰因果检验
#                     test_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
#                     # 提取所有滞后阶数的p值
#                     p_values = [test_result[lag + 1][0]['ssr_ftest'][1] for lag in range(max_lag)]
#                     # 取最小p值作为因果关系的显著性
#                     min_p_value = min(p_values)
#                     causality_matrices[batch_idx, i, j] = min_p_value
#
#     partition = 0.8
#     causality_matrices = matrices_sparsification(matrices=causality_matrices, partition=partition)
#     return causality_matrices


# 矩阵稀疏化
def matrices_sparsification(matrices, partition):
    b, N, _ = matrices.shape  # 获取批处理大小和矩阵大小
    result = np.zeros((b, N, N), dtype=int)  # 初始化结果矩阵为全零

    for i in range(b):  # 遍历批处理中的每个矩阵
        for j in range(N):  # 遍历矩阵的每一行
            # 获取当前行的数据，并找到partition*N个最小的元素的索引
            indices = np.argpartition(matrices[i, j], partition * N)[:partition * N]
            # 将这些索引位置的元素设置为1
            result[i, j, indices] = 1

    return result

def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    # Remove errors for time steps when transition to new channel (as this will be impossible for model to predict)
    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'./datasets/data/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv('./datasets/data/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    # Sort values by channel
    md = md.sort_values(by=['chan_id'])

    # Getting the cumulative start index for each channel
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    # Normalize each concatenated part individually
    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        # e_s = adjusted_scores[c_start: c_end+1]
        # e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
        # adjusted_scores[c_start: c_end+1] = e_s
        if c_start >= c_end or c_start >= len(adjusted_scores) or c_end > len(adjusted_scores):
            print(f"Warning: Invalid indices c_start={c_start}, c_end={c_end}. Skipping this segment.")
            continue

        e_s = adjusted_scores[c_start: c_end + 1]
        if len(e_s) == 0:
            print(f"Warning: Empty segment at c_start={c_start}, c_end={c_end}. Skipping normalization.")
            continue

        e_s = (e_s - np.min(e_s)) / (np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end + 1] = e_s

    return adjusted_scores