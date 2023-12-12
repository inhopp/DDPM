import os
import torch
import torch.nn as nn
import torch.optim as optim
import diffusion
from data import generate_loader
from option import get_option
from model import UNet
from tqdm import tqdm
from diffusion import p_losses, sample
from torchvision.utils import save_image

class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.dev = torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.model = UNet(dim=opt.input_size, channels=opt.input_channels, dim_mults=(1,2,4,)).to(self.dev)

        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, opt.data_name, "best_epoch.pt")
            self.model.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.model = nn.DataParallel(self.model, device_ids=self.opt.device_ids).to(self.dev)

        print("# params:", sum(map(lambda x: x.numel(), self.model.parameters())))

        # self.loss_fn = p_losses
        self.optim = optim.Adam(self.model.parameters(), lr=opt.lr)

        self.train_loader = generate_loader('train', opt)
        print("train set ready")
        self.best_score, self.best_epoch = 0, 0


    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            self.model.train()
            loop = tqdm(self.train_loader)

            for batch, (images, _) in enumerate(loop):
                batch_size = images.shape[0]
                images = images.to(self.dev)

                t = torch.randint(0, diffusion.timesteps, (batch_size,), device=self.dev).long()

                loss = p_losses(self.model, images, t)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()   

                if (batch % 100 == 0):
                    print("Loss:", loss.item())


                if (epoch+1) % 24 == 0:
                    generated_imgs = sample(self.model)
                    save_image(generated_imgs[:25], f"data{epoch}.png", nrow=5, normalize=True)


    
    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root, self.opt.data_name), exist_ok=True)
        save_path = os.path.join(self.opt.ckpt_root, self.opt.data_name, "best_epoch.pt")
        torch.save(self.model.state_dict(), save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()

if __name__ == "__main__":
    main()