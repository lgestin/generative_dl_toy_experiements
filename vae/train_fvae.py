import torch

from pathlib import Path
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

from data.dataset import MNIST

from vae.fvae.fvae import FVAE


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

    vae = FVAE(d_model=d_model)
    vae = vae.to(device)

    opt = torch.optim.AdamW(vae.parameters(), lr=lr)

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

        x_vae, log_det, (z_q, mu_q, log_std_q), (z_p, mu_p, log_std_p) = model(x, y)

        # Train VAE
        loss_recons = (x - x_vae).pow(2).mean()

        loss_kl = (
            log_std_p
            - log_std_q
            - 0.5
            + 0.5 * (z_p - mu_p).pow(2) * (-2 * log_std_p).exp()
            - 0.5 * (z_q - mu_q).pow(2) * (-2 * log_std_q).exp()
        )
        loss_kl = loss_kl.mean()

        loss = loss_recons + 1e0 * loss_kl - 1e0 * log_det.mean()

        with torch.no_grad():
            # Compute kl p | N(0, 1)
            loss_klp = -log_std_p + 0.5 * ((2 * log_std_p).exp() + mu_p.pow(2) - 1)
            # Compute kl q | N(0, 1)
            loss_klq = -log_std_q + 0.5 * ((2 * log_std_q.exp()) + mu_q.pow(2) - 1)
            loss_klp, loss_klq = loss_klp.mean(), loss_klq.mean()

        if model.training:
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()

        metrics = {
            "loss": loss,
            "recons": loss_recons,
            "kl": loss_kl,
            "klp": loss_klp,
            "klq": loss_klq,
            "mu_p": mu_p.mean(),
            "std_p": log_std_p.exp().mean(),
            "std_q": log_std_q.exp().mean(),
            "mu_q": mu_q.mean(),
            "log_det": log_det.mean(),
        }
        if model.training:
            metrics["grad"] = grad

        return metrics

    torch.backends.cudnn.benchmark = True
    while 1:
        for batch in dataloader:
            step += 1

            vae.train()
            metrics = process_batch(batch, vae)

            # Monitor metrics
            for k, v in metrics.items():
                writer.add_scalar(f"train/{k}", v, global_step=step)

            if step % 1000 == 1:
                val_metrics = defaultdict(list)
                vae.eval()
                for batch in val_dataloader:
                    with torch.no_grad():
                        metrics = process_batch(batch, vae)
                    for k, v in metrics.items():
                        val_metrics[k] += [v]

                for k, v in val_metrics.items():
                    writer.add_scalar(
                        f"val/{k}", torch.stack(v).mean(), global_step=step
                    )

                x_vae, _, _, _ = vae(x_val, y_val)
                x_vae = (MNIST.unnormalize(x_vae) + 1) / 2
                writer.add_images("val/rec", x_vae, global_step=step)

                x_dec = vae.sample(z)
                x_dec = (MNIST.unnormalize(x_dec) + 1) / 2
                writer.add_images("val/gen", x_dec, global_step=step)

            # Save models
            if step % 10000 == 1:
                torch.save(vae.state_dict(), exp_path / f"vae_{step}.pt")

            if step > max_iter:
                return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_path", type=Path, required=True)
    parser.add_argument("--data_path", type=Path, default=Path.home() / ".data/mnist")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iter", type=int, default=200000)

    options = parser.parse_args()

    main(
        exp_path=options.exp_path,
        data_path=options.data_path,
        d_model=options.d_model,
        lr=options.lr,
        batch_size=options.batch_size,
        max_iter=options.max_iter,
    )
