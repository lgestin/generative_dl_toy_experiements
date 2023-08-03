import torch

from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from data.dataset import MNIST

from normalizing_flows.maf.maf import MAF


def main(
    exp_path: Path,
    data_path: Path,
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

    maf = MAF(d_model=28 * 28, n_layers=5)
    maf = maf.to(device)

    opt = torch.optim.AdamW(maf.parameters(), lr=lr)

    writer = SummaryWriter(exp_path)
    x_val = next(iter(val_dataloader))[0][:16]
    x_val = x_val.to(device)
    writer.add_images("val/input", (MNIST.unnormalize(x_val) + 1) / 2)

    val_metrics = defaultdict(list)
    step = 0

    def process_batch(batch, model):
        metrics = {}

        x, _ = batch
        x = x.view(x.size(0), 28 * 28).to(device)

        z, log_det = model(x)

        # Train MAF
        nll = 0.5 * (x.size(1) * torch.tensor(2 * torch.pi).log() + z.pow(2).sum(dim=1))
        loss = nll.mean() - log_det.mean()

        metrics = {"loss": loss, "nll": nll.mean(), "log_det": log_det.mean()}
        if model.training:
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()

            metrics["grad"] = grad
        return metrics

    torch.backends.cudnn.benchmark = True
    while 1:
        for batch in dataloader:
            step += 1

            maf.train()
            metrics = process_batch(batch, maf)

            # Monitor metrics
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step=step)

            if step % 1000 == 1:
                val_metrics = defaultdict(list)
                maf.eval()
                for batch in val_dataloader:
                    with torch.no_grad():
                        metrics = process_batch(batch, maf)
                    for k, v in metrics.items():
                        val_metrics[k] += [v]

                for k, v in val_metrics.items():
                    writer.add_scalar(
                        f"val/{k}", torch.stack(v).mean(), global_step=step
                    )

                z = torch.randn(16, 28 * 28).to(device)
                with torch.no_grad():
                    x_maf, _ = maf.inverse(z)
                x_maf = (MNIST.unnormalize(x_maf) + 1) / 2
                writer.add_images(
                    "val/gen", x_maf.view(16, 1, 28, 28), global_step=step
                )

            # Save models
            if step % 10000 == 1:
                torch.save(maf, exp_path / f"maf_{step}.pt")

            if step > max_iter:
                return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path.home() / ".data/mnist")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_iter", type=int, default=100000)

    options = parser.parse_args()

    main(
        exp_path=options.exp_path,
        data_path=options.data_path,
        lr=options.lr,
        batch_size=options.batch_size,
        max_iter=options.max_iter,
    )
