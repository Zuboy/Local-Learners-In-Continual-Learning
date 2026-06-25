import numpy as np
import torch

from models.gem import store_grad, overwrite_grad, project2cone2
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, fill_buffer
from utils.conf import warn_once


class LocalGem(ContinualModel):
    """Local learning variant of GEM: gradient projection applied per local head."""
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

        # params[0] = fc1 + local_head1, params[1] = fc2 + local_head2
        self.params = [
            list(self.net.fc1.parameters()) + list(self.net.local_head1.parameters()),
            list(self.net.fc2.parameters()) + list(self.net.local_head2.parameters()),
        ]
        self.opt1 = self.get_optimizer(self.params[0])
        self.opt2 = self.get_optimizer(self.params[1])
        self.head_optimizers = [self.opt1, self.opt2]

        # per-head: number of scalars per parameter tensor
        self.grad_dims = [[p.data.numel() for p in pg] for pg in self.params]

        # per-head per-task memory gradient storage; grows by one entry in end_task
        self.grads_cs = [[], []]

        # per-head current-data gradient buffer
        self.grads_da = [
            torch.zeros(sum(self.grad_dims[0])).to(self.device),
            torch.zeros(sum(self.grad_dims[1])).to(self.device),
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
        # allocate gradient storage for the task that just ended
        for head_id in range(2):
            self.grads_cs[head_id].append(
                torch.zeros(sum(self.grad_dims[head_id])).to(self.device)
            )
        fill_buffer(self.buffer, dataset, self.current_task,
                    required_attributes=['examples', 'labels', 'task_labels'])

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_task_labels = self.buffer.get_data(
                self.args.buffer_size, transform=self.transform, device=self.device)

            for tt in buf_task_labels.unique():
                for opt in self.head_optimizers:
                    opt.zero_grad()

                cur_task_inputs = buf_inputs[buf_task_labels == tt]
                cur_task_labels = buf_labels[buf_task_labels == tt]

                # local_forward detaches h1 before fc2, so loss1/loss2 have disjoint graphs
                out1, out2 = self.net.local_forward(cur_task_inputs)
                self.loss(out1, cur_task_labels).backward()
                self.loss(out2, cur_task_labels).backward()

                for h in range(2):
                    store_grad(lambda h=h: iter(self.params[h]),
                               self.grads_cs[h][tt], self.grad_dims[h])

        # current-data gradients
        for opt in self.head_optimizers:
            opt.zero_grad()

        out1, out2 = self.net.local_forward(inputs)
        loss1 = self.loss(out1, labels)
        loss2 = self.loss(out2, labels)
        loss1.backward()
        loss2.backward()

        if not self.buffer.is_empty():
            for h in range(2):
                store_grad(lambda h=h: iter(self.params[h]),
                           self.grads_da[h], self.grad_dims[h])

            for h in range(2):
                dot_prod = torch.mm(
                    self.grads_da[h].unsqueeze(0),
                    torch.stack(self.grads_cs[h]).T
                )
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self.solver, self.grads_da[h].unsqueeze(1),
                                  torch.stack(self.grads_cs[h]).T,
                                  margin=self.args.gamma)
                    overwrite_grad(lambda h=h: iter(self.params[h]),
                                   self.grads_da[h], self.grad_dims[h])

        for opt in self.head_optimizers:
            opt.step()

        return loss1.item() + loss2.item()
