import torch

from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from data.dataset import MNIST

from diffusion.conditional.model import DiffusionModel


def main(
    exp_path: Path,
    data_path: Path,
    d_model: int,
    lr: float,
    batch_size: int,
    max_iter: int,
):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    dataset = MNIST(data_path=data_path)

    val_idx = list(range(1000))
    val_dataset = Subset(dataset, val_idx)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    test_idx = list(range(1000, 2000))
    test_dataset = Subset(dataset, test_idx)

    train_idx = list(range(2000, len(dataset)))
    train_dataset = Subset(dataset, train_idx)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    T = 50
    model = DiffusionModel(d_model=d_model, beta_0=0.001, beta_T=0.3, T=T)
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    writer = SummaryWriter(exp_path)
    x_val, y_val = next(iter(val_dataloader))
    x_val, y_val = x_val[:16].to(device), y_val[:16].to(device)
    writer.add_images("val/input", (MNIST.unnormalize(x_val) + 1) / 2)

    z = torch.cat([torch.arange(10), 10 * torch.ones((6,))]).long().to(device)

    val_metrics = defaultdict(list)
    step = 0

    def process_batch(batch, model):
        metrics = {}

        x, y = batch
        x, y = x.to(device), y.to(device)

        # randomly drop some of the labels
        mask = torch.rand_like(y.float()) > 0.7
        y = y.masked_fill(mask, 10)

        t = torch.randint(0, T, (x.size(0),), device=device)
        x_t, z_t = model.forward_process(x, t)
        z_pred_t = model.reverse_process(x_t, z_cond=y, t=t)

        # Train model
        loss = (z_t - z_pred_t).pow(2).mean()

        if model.training:
            opt.zero_grad()
            loss.backward()
            opt.step()

        metrics = {"loss": loss}
        return metrics

    while 1:
        for batch in dataloader:
            step += 1

            model.train()
            metrics = process_batch(batch, model)

            # Monitor metrics
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step=step)

            if step % 5000 == 1:
                val_metrics = defaultdict(list)
                model.eval()
                for batch in val_dataloader:
                    with torch.no_grad():
                        metrics = process_batch(batch, model)
                    for k, v in metrics.items():
                        val_metrics[k] += [v]

                for k, v in val_metrics.items():
                    writer.add_scalar(
                        f"val/{k}", torch.stack(v).mean(), global_step=step
                    )

                x_T = torch.randn(16, 1, 28, 28).to(device)
                x_gen, x_ts = model.denoise(x_T, z_cond=z)
                x_gen = (MNIST.unnormalize(x_gen) + 1) / 2
                writer.add_images("val/gen", x_gen, global_step=step)
                writer.add_images(
                    "val/x_ts", x_ts[:1].transpose(1, 0), global_step=step
                )

            # Save models
            if step % 10000 == 1:
                torch.save(model.state_dict(), exp_path / f"dm_{step}.pt")

            if step > max_iter:
                return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path.home() / "/.data/mnist")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
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
