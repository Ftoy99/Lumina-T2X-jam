import torch as th
from torchdiffeq import odeint


def sample(x1, mf1):
    """Sampling x0 & t based on shape of x1 (if needed)
    Args:
      x1 - data point; [batch, *dim]
    """
    if isinstance(x1, (list, tuple)):
        x0 = [th.randn_like(img_start) for img_start in x1]
        mf0 = [th.randn_like(img_start) for img_start in mf1]
    else:
        x0 = th.randn_like(x1)
        mf0 = th.randn_like(mf1)

    t = th.rand((len(x1),))
    t = t.to(x1[0])
    return t, x0, x1, mf0, mf1


def training_losses(model, x1, mf1, model_kwargs=None):
    """Loss for training the score model
    Args:
    - model: backbone model; could be score, noise, or velocity
    - x1: datapoint
    - model_kwargs: additional arguments for the model
    """
    if model_kwargs == None:
        model_kwargs = {}

    B = len(x1)

    t, x0, x1, mf0, mf1 = sample(x1, mf1)

    if isinstance(x1, (list, tuple)):
        xt = [t[i] * x1[i] + (1 - t[i]) * x0[i] for i in range(B)]
        ut = [x1[i] - x0[i] for i in range(B)]

        mft = [t[i] * mf1[i] + (1 - t[i]) * mf0[i] for i in range(B)]
        mfut = [mf1[i] - mf0[i] for i in range(B)]
    else:
        dims = [1] * (len(x1.size()) - 1)
        t_ = t.view(t.size(0), *dims)

        xt = t_ * x1 + (1 - t_) * x0
        ut = x1 - x0

        mft = t_ * mf1 + (1 - t_) * mf0
        mfut = mf1 - mf0

    model_output = model(xt, mft, t, **model_kwargs)

    terms = {}

    if isinstance(x1, (list, tuple)):
        terms["loss"] = th.stack(
            [((ut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        )
        terms["mf_loss"] = th.stack(
            [((mfut[i] - model_output[i]) ** 2).mean() for i in range(B)],
            dim=0,
        )
    else:
        terms["loss"] = ((model_output - ut) ** 2).mean(dim=list(range(1, ut.ndim)))
        terms["mf_loss"] = ((model_output - mfut) ** 2).mean(dim=list(range(1, mfut.ndim)))

    return terms


class ODE:
    """ODE solver class"""

    def __init__(
            self,
            num_steps,
            sampler_type="euler",
            time_shifting_factor=None,
            t0=0.0,
            t1=1.0,
            use_sd3=False,
            strength=1.0,
    ):
        if use_sd3:
            self.t = th.linspace(t1, t0, num_steps)
            if time_shifting_factor:
                self.t = (time_shifting_factor * self.t) / (1 + (time_shifting_factor - 1) * self.t)
        else:
            self.t = th.linspace(t0, t1, num_steps)
            if time_shifting_factor:
                self.t = self.t / (self.t + time_shifting_factor - time_shifting_factor * self.t)

        if strength != 1.0:
            self.t = self.t[int(num_steps * (1 - strength)):]

        self.use_sd3 = use_sd3
        self.sampler_type = sampler_type

    def sample(self, x, xmf, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device
        print("in ode sample")

        def _fn(t, x, xmf):
            t = th.ones(x[0].size(0)).to(device) * t if isinstance(x, tuple) else th.ones(x.size(0)).to(device) * t
            model_output, model_output_xmf = model(x, xmf, t, **model_kwargs)
            return model_output

        t = self.t.to(device)
        # samples = odeint(_fn, x, t, method=self.sampler_type)
        # https://scicomp.stackexchange.com/questions/41905/passing-additional-arguments-to-odeint-from-torchdiffeq-to-solve-an-ivp

        # sol = odeint(f, x0, t, method='euler', args=(omega,))
        # sol = odeint(lambda y, t: f(y, t, omega), x0, t, method='euler')
        # Samples
        # samples_x = odeint(lambda y, t: _fn(y, t, xmf), x, t, method=self.sampler_type)  # holy shit
        # Use Midpoint method for integration
        samples_x = x
        samples_xmf = xmf
        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]

            # First stage (evaluate at the current time)
            model_output, model_output_xmf = _fn(t[i], samples_x, samples_xmf)
            k1_x = model_output
            k1_xmf = model_output_xmf

            # Second stage (evaluate at the midpoint)
            midpoint_x = samples_x + 0.5 * k1_x * dt
            midpoint_xmf = samples_xmf + 0.5 * k1_xmf * dt
            model_output, model_output_xmf = _fn(t[i] + 0.5 * dt, midpoint_x, midpoint_xmf)
            k2_x = model_output
            k2_xmf = model_output_xmf

            # Update the state using the midpoint approximation
            samples_x = samples_x + k2_x * dt
            samples_xmf = samples_xmf + k2_xmf * dt
        return samples_x, samples_xmf
