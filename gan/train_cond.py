import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MNIST

from gan.conditional.generator import Generator
from gan.conditional.discriminator import Discriminator


def main(
    exp_path: Path,
    data_path: Path,
    d_model: int,
    lr: float,
    batch_size: int,
    max_iter: int,
):
    cuda = torch.cuda.is_available()

    dataset = MNIST(data_path=data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    gen = Generator(d_z=(1, 4, 4), d_model=d_model)
    dis = Discriminator(d_z=(1, 4, 4), d_model=d_model)

    if cuda:
        gen = gen.cuda()
        dis = dis.cuda()

    gopt = torch.optim.AdamW(gen.parameters(), lr=3 * lr, betas=(0.5, 0.9))
    dopt = torch.optim.AdamW(dis.parameters(), lr=lr, betas=(0.5, 0.9))

    writer = SummaryWriter(exp_path)
    writer.add_image("data", (MNIST.unnormalize(dataset[0][0]) + 1) / 2, global_step=0)
    y_test = torch.cat([torch.arange(10), 10 * torch.ones((6,))]).long()
    if cuda:
        y_test = y_test.cuda()
    labels = " ".join([str(l) for l in y_test[:16].tolist()])
    writer.add_text("train/imgs_labels", labels)

    step = 0
    while 1:
        for x, y in dataloader:
            step += 1

            gen.train()
            if cuda:
                x, y = x.cuda(), y.cuda()
                mask = torch.rand_like(y.float()) > 0.70
                y = y.masked_fill(mask, 10)

            x_gen, z = gen.sample(z_cond=y)

            # Train discriminator
            y_dis_real = dis(x, z_cond=y)
            y_dis_fake = dis(x_gen.detach(), z_cond=y)

            loss_dis_r = F.relu(1 - y_dis_real)
            loss_dis_f = F.relu(1 + y_dis_fake)
            loss_dis = 0.5 * (loss_dis_r + loss_dis_f).mean()

            dopt.zero_grad()
            loss_dis.backward()
            dopt.step()

            # Train generator
            y_dis_fake = dis(x_gen, z_cond=y)

            loss_gen = -y_dis_fake.mean()

            gopt.zero_grad()
            loss_gen.backward()
            gopt.step()

            # Monitor metrics
            writer.add_scalar("train/gen", loss_gen, global_step=step)
            writer.add_scalar("train/dis", loss_dis, global_step=step)
            writer.add_scalar("train/dis_r", loss_dis_r.mean(), global_step=step)
            writer.add_scalar("train/dis_f", loss_dis_f.mean(), global_step=step)

            if step % 5000 == 1:
                gen.eval()
                with torch.no_grad():
                    x_gen, _ = gen.sample(z_cond=y_test)
                x_gen = (MNIST.unnormalize(x_gen) + 1) / 2
                writer.add_images("train/imgs", x_gen[:16], global_step=step)

            # Save models
            if step % 10000 == 0:
                torch.save(gen.state_dict(), exp_path / f"gen_{step}.pt")
                torch.save(dis.state_dict(), exp_path / f"dis_{step}.pt")

            if step > max_iter:
                return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path.home() / "/.data/mnist")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=150000)

    options = parser.parse_args()

    main(
        exp_path=options.exp_path,
        data_path=options.data_path,
        d_model=options.d_model,
        lr=options.lr,
        batch_size=options.batch_size,
        max_iter=options.max_iter,
    )
