"""Drop-in TensorBoard SummaryWriter wrapper that mirrors `add_scalar` to wandb.

Trainers do `self.logger = SummaryWriter(opt.log_dir)` today. Swap that for
`self.logger = make_logger(opt)` and wandb logging happens transparently when
`opt.use_wandb` is set; otherwise the wrapper is a thin pass-through to the
underlying SummaryWriter (zero behavioral change).

Wandb step is left to the library's auto counter — different add_scalar call
sites use different step bases (iteration for Train/*, epoch for Val/*), so
we don't pass `step=` to avoid wandb's monotonic-step requirement.
"""
from torch.utils.tensorboard import SummaryWriter


class _TBWandbLogger:
    def __init__(self, opt):
        self._tb = SummaryWriter(opt.log_dir)
        self._wb = None
        if not getattr(opt, "use_wandb", False):
            return

        import wandb
        config = {
            k: v for k, v in vars(opt).items()
            if isinstance(v, (int, float, str, bool, list, tuple))
        }
        tags = []
        team = getattr(opt, "wandb_team", "") or ""
        if team:
            tags.append(f"team:{team}")

        self._wb = wandb.init(
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            name=(getattr(opt, "wandb_run_name", "") or opt.name),
            config=config,
            tags=tags,
            dir=opt.log_dir,
        )

    def add_scalar(self, tag, value, step=None):
        self._tb.add_scalar(tag, value, step)
        if self._wb is not None:
            self._wb.log({tag: float(value)})

    def close(self):
        self._tb.close()
        if self._wb is not None:
            self._wb.finish()

    def __getattr__(self, name):
        return getattr(self._tb, name)


def make_logger(opt):
    return _TBWandbLogger(opt)
