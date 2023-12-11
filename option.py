import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--multigpu", type=bool, default=True)
    parser.add_argument("--device", type=str, default="0")

    # models
    parser.add_argument("--pretrained", type=bool, default=False)

    # dataset
    parser.add_argument("--data_dir", type=str, default="./datasets/")
    parser.add_argument("--data_name", type=str, default="ffhq")

    # training setting
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--n_epoch", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)

    # misc
    parser.add_argument("--ckpt_root", type=str, default="./FT_model")

    return parser.parse_args()


def make_template(opt):
    # device
    opt.device_ids = [int(item) for item in opt.device.split(',')]
    if len(opt.device_ids) == 1:
        opt.multigpu = False
    opt.gpu = opt.device_ids[0]


def get_option():
    opt = parse_args()
    make_template(opt)
    return opt