import torch
import torch.nn.functional as F

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from data.dataset import MNIST

from gan.unconditional.generator import UnconditionalGenerator
from gan.unconditional.discriminator import Discriminator


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

    gen = UnconditionalGenerator(d_z=(1, 4, 4), d_model=d_model)
    dis = Discriminator(d_model=d_model)

    if cuda:
        gen = gen.cuda()
        dis = dis.cuda()

    gopt = torch.optim.AdamW(gen.parameters(), lr=3 * lr, betas=(0.5, 0.9))
    dopt = torch.optim.AdamW(dis.parameters(), lr=lr, betas=(0.5, 0.9))

    writer = SummaryWriter(exp_path)
    writer.add_image("test", (MNIST.unnormalize(dataset[0][0]) + 1) / 2, global_step=0)

    step = 0
    while 1:
        for x, _ in dataloader:
            step += 1
            if cuda:
                x = x.cuda()

            x_gen, z = gen.sample(batch_size=x.size(0))

            # Train discriminator
            y_dis_real = dis(x)
            y_dis_fake = dis(x_gen.detach())

            loss_dis_r = F.relu(1 - y_dis_real)
            loss_dis_f = F.relu(1 + y_dis_fake)
            loss_dis = 0.5 * (loss_dis_r + loss_dis_f).mean()

            dopt.zero_grad()
            loss_dis.backward()
            dopt.step()

            # Train generator
            y_dis_fake = dis(x_gen)

            loss_gen = -y_dis_fake.mean()

            gopt.zero_grad()
            loss_gen.backward()
            gopt.step()

            # Monitor metrics
            writer.add_scalar("train/gen", loss_gen, global_step=step)
            writer.add_scalar("train/dis", loss_dis, global_step=step)
            writer.add_scalar("train/dis_r", loss_dis_r.mean(), global_step=step)
            writer.add_scalar("train/dis_f", loss_dis_f.mean(), global_step=step)

            if step % 10000 == 1:
                x_gen = (MNIST.unnormalize(x_gen) + 1) / 2
                writer.add_images("train/imgs", x_gen[:16], global_step=step)

            # Save models
            if step % 100000 == 1:
                torch.save(gen, exp_path / f"gen_{step}.pt")
                torch.save(dis, exp_path / f"dis_{step}.pt")

            if step > max_iter:
                break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument(
        "--data_path", type=Path, default=Path("/home/lucas/.data/mnist")
    )
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iter", type=int, default=100000)

    options = parser.parse_args()

    main(
        exp_path=options.exp_path,
        data_path=options.data_path,
        d_model=options.d_model,
        lr=options.lr,
        batch_size=options.batch_size,
        max_iter=options.max_iter,
    )
