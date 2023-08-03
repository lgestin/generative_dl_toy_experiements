from vae.unconditional.vae_uncond import VAE
from vae.conditional.conditional_prior import ConditionalPrior


class ConditionalVAE(VAE):
    def __init__(self, d_model):
        super().__init__(d_model)
        self.prior = ConditionalPrior(d_model)

    def forward(self, x, z_cond):
        mu_p, log_std_p = self.encoder(x)

        z = self.sample_z(x.size(0))
        z = z * log_std_p.exp() + mu_p

        mu_q, log_std_q = self.prior(z_cond)

        return self.decoder(z), (mu_p, log_std_p), (mu_q, log_std_q)

    def sample(self, z_cond):
        mu_q, log_std_q = self.prior(z_cond)

        z = self.sample_z(z_cond.size(0))
        z = z * log_std_q.exp() + mu_q

        return self.decoder(z)
