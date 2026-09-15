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
        assert hasattr(self.net, 'blocks') and hasattr(self.net, 'heads')
        self.local_optimizers = [
            self.get_optimizer(list(block.parameters()) + list(head.parameters()))
            for block, head in zip(self.net.blocks, self.net.heads)
        ]

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
        outputs = self.net.forward_all_heads(inputs)

        total_loss = 0.0
        for out, optimizer in zip(outputs, self.local_optimizers):
            optimizer.zero_grad()
            loss = self.loss(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size]
        )

        return total_loss
