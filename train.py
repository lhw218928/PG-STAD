import json

from datetime import datetime
import torch.nn as nn

from model import PT_STAD
from args import get_parser
from training import Trainer
from utils import *
from prediction import Predictor
if __name__ == "__main__":

    id = datetime.now().strftime("%d%m%Y_%H%M%S")

    parser = get_parser()
    args = parser.parse_args()

    # 参数
    dataset = args.dataset
    n_epochs = args.epochs
    window_size = args.window_size
    window_num = args.window_num
    spec_res = args.spec_res
    normalize = args.normalize
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP', 'SWAT']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        # raise Exception(f'Dataset "{dataset}" not available.')
        print('Invalid dataset')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    x_train = torch.from_numpy(x_train).float()
    x_test = torch.from_numpy(x_test).float()
    features_num = x_train.shape[1]

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        target_dims = features_num
        print(f"Will forecast and reconstruct all {features_num} input features with size: {window_size}")
    else:
        print(f"Will forecast and reconstruct  {target_dims} input features with size: {window_size}:")

    #horizon决定执行单步还是多步预测
    train_dataset = SlidingWindowDataset(data = x_train, window_size = window_size, window_num = window_num, target_dim = target_dims)
    test_dataset = SlidingWindowDataset(data = x_test, window_size = window_size, window_num = window_num, target_dim = target_dims)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split, shuffle_dataset, test_dataset=test_dataset
    )

    model = PT_STAD(
        features_num,
        window_size,
        window_num,
        target_dims,
        structure_feature_embed_dim = None,
        use_gatv2 = args.use_gatv2,
        gru_layers = args.gru_n_layers,
        time_feature_embed_dim = None,
        forecast_hidden_dim=args.fc_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        recon_n_layers=args.recon_n_layers,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        features_num,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SWAT": (0.90, 0.005),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    level, q = level_q_dict[key]
    if args.level is not None:
        level = args.level
    if args.q is not None:
        q = args.q

    # Some suggestions for Epsilon args
    reg_level_dict = {"SMAP": 0, "MSL": 0,"SWAT":0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        "level": level,
        "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    predictor = Predictor(
        best_model,
        window_size,
        window_num,
        features_num,
        prediction_args,
    )

    train_max_idx = (len(train_dataset) + window_num ) * window_size
    test_max_idx = (len(test_dataset) + window_num) * window_size
    label = y_test[window_size * window_num: test_max_idx] if y_test is not None else None
    predictor.predict_anomalies(x_train[:train_max_idx], x_test[:test_max_idx], label)

    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)