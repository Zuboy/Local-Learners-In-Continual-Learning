import numpy as np
import torch

from models.gem import store_grad, overwrite_grad, project2cone2
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from utils.conf import warn_once


class LocalGem(ContinualModel):
    """26.06.26 Local learning variant of GEM: gradient projection applied per local head."""
    NAME = 'local-gem'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--gamma', type=float, default=0.5,
                            help='Margin parameter for GEM.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.buffer = Buffer(self.args.buffer_size)

        # param group + optimizer per (block, head) pair, built in one loop
        assert hasattr(self.net, 'blocks') and hasattr(self.net, 'heads'), \
            "local-gem needs a local backbone exposing `blocks` and `heads` (e.g. local-mnistmlp)."
        self.params, self.head_optimizers = [], []
        for block, head in zip(self.net.blocks, self.net.heads):
            param_group = list(block.parameters()) + list(head.parameters())
            self.params.append(param_group)
            self.head_optimizers.append(self.get_optimizer(param_group))

        self.n_heads = len(self.params)

        # per-head storage: flat past-task gradients + current gradient
        self.grad_dims = [[p.data.numel() for p in pg] for pg in self.params]
        self.grads_cs = [[] for _ in range(self.n_heads)]
        self.grads_da = [
            torch.zeros(sum(dims)).to(self.device) for dims in self.grad_dims
        ]

        try:
            import quadprog as solver  # type: ignore
        except ImportError:
            warn_once("`quadprog` not found, trying with `qpsolvers`. Note that the code is only tested with `quadprog`.")
            try:
                import qpsolvers as solver  # type: ignore
                raise Exception('QPSolvers is just a suggestion but does not work at the moment. To make it work, you need to set it up properly (and remove this exception).')
            except ImportError:
                raise Exception('GEM requires quadprog (linux only, python <= 3.10) or qpsolvers (cross-platform)')
        self.solver = solver

    def end_task(self, dataset):
        for head_id in range(self.n_heads):
            self.grads_cs[head_id].append(
                torch.zeros(sum(self.grad_dims[head_id])).to(self.device)
            )

        examples_per_task = self.args.buffer_size // (self.current_task + 1)

        if len(self.buffer) > 0:
            buf_x, buf_y, buf_tl = self.buffer.get_all_data()
            self.buffer.empty()
            for tt in buf_tl.unique():
                idx = (buf_tl == tt)
                self.buffer.add_data(
                    examples=buf_x[idx][:examples_per_task],
                    labels=buf_y[idx][:examples_per_task],
                    task_labels=buf_tl[idx][:examples_per_task])

        counter = 0
        for data in dataset.train_loader:
            y, not_aug_x = data[1], data[2]
            if counter >= examples_per_task:
                break
            take = min(examples_per_task - counter, not_aug_x.shape[0])
            self.buffer.add_data(
                examples=not_aug_x[:take],
                labels=y[:take],
                task_labels=torch.full((take,), self.current_task, dtype=torch.long))
            counter += take

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        has_memory = not self.buffer.is_empty()

        # past-task gradients, one flat vector per (task, head)
        if has_memory:
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.buffer_size, transform=self.transform, device=self.device)

            for tt in buf_task_labels.unique():  # seen task
                cur_task_inputs = buf_inputs[buf_task_labels == tt]
                cur_task_labels = buf_labels[buf_task_labels == tt]
                outs = self.net.local_forward(cur_task_inputs)

                for h, (out, optimizer) in enumerate(zip(outs, self.head_optimizers)):
                    optimizer.zero_grad()
                    self.loss(out, cur_task_labels).backward()
                    store_grad(lambda h=h: iter(self.params[h]),
                               self.grads_cs[h][tt], self.grad_dims[h])

        # current data: per head -> backward, project if it conflicts, step
        outs = self.net.local_forward(inputs)

        total_loss = 0.0
        for h, (out, optimizer) in enumerate(zip(outs, self.head_optimizers)):
            optimizer.zero_grad()
            loss = self.loss(out, labels)
            loss.backward()
            total_loss += loss.item()

            if has_memory:
                store_grad(lambda h=h: iter(self.params[h]),
                           self.grads_da[h], self.grad_dims[h])

                memories = torch.stack(self.grads_cs[h]).T
                dot_prod = torch.mm(self.grads_da[h].unsqueeze(0), memories)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self.solver, self.grads_da[h].unsqueeze(1),
                                  memories, margin=self.args.gamma)
                    overwrite_grad(lambda h=h: iter(self.params[h]),
                                   self.grads_da[h], self.grad_dims[h])

            optimizer.step()

        return total_loss
