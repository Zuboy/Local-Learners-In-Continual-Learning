import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


class LocalEr(ContinualModel):
    NAME = 'local-er'
    COMPATIBILITY = ['class-il', 'task-il', 'domain-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)
        self.self_opt()

    def self_opt(self):
        self.opt1 = self.get_optimizer(
            list(self.net.fc1.parameters()) +
            list(self.net.local_head1.parameters())
        )
        self.opt2 = self.get_optimizer(
            list(self.net.fc2.parameters()) +
            list(self.net.local_head2.parameters())
        )
        self.local_optimizers = [self.opt1, self.opt2]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        real_batch_size = inputs.shape[0]

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform,
                device=self.device
            )

            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        for optimizer in self.local_optimizers:
            optimizer.zero_grad()

        out1, out2 = self.net.local_forward(inputs)
        loss1 = self.loss(out1, labels)
        loss2 = self.loss(out2, labels)

        loss1.backward()
        self.opt1.step()

        loss2.backward()
        self.opt2.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size]
        )

        total_loss = loss1.item() + loss2.item()

        return total_loss
