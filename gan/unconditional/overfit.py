import torch

from torch.utils.tensorboard import SummaryWriter

from data.dataset import MNIST
from gan.unconditional.generator import UnconditionalGenerator


def main():
    dataset = MNIST(data_path="/home/lucas/.data/mnist")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)

    gen = UnconditionalGenerator(d_z=(1, 4, 4), d_model=32).cuda()
    opt = torch.optim.AdamW(gen.parameters(), lr=3e-4)

    writer = SummaryWriter()
    writer.add_image("test", (MNIST.unnormalize(dataset[0][0]) + 1) / 2, global_step=0)

    batch = next(iter(dataloader))[0].cuda()
    for i in range(100000):
        x_gen = gen.sample(32)
        loss = (batch - x_gen).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)

        if i % 200 == 0:
            writer.add_image(
                "pred", (MNIST.unnormalize(x_gen[0]) + 1) / 2, global_step=i
            )


if __name__ == "__main__":
    main()
