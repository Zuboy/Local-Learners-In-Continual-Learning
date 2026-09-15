import torch
import torch.nn.functional as F

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
from utils.buffer import Buffer


class LocalDerpp(ContinualModel):
    NAME = 'local-derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--buffer_size', type=int, required=True)
        parser.add_argument('--minibatch_size', type=int, required=True)
        parser.add_argument('--alpha', type=float, default=0.5)
        parser.add_argument('--beta', type=float, default=0.5)
        parser.add_argument('--local_loss_weight', type=float, default=1.0)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size, self.device)

        # one optimizer per (block, head) pair -> any number of local layers
        assert hasattr(self.net, 'blocks') and hasattr(self.net, 'heads'), \
            "local-derpp needs a local backbone exposing `blocks` and `heads` (e.g. local-mnistmlp)."
        self.head_optimizers = [
            self.get_optimizer(list(block.parameters()) + list(head.parameters()))
            for block, head in zip(self.net.blocks, self.net.heads)
        ]

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        outputs_all = self.net.forward_all_heads(inputs)
        n_heads = len(outputs_all)

        # replay forward once for all heads (each head distills its own logits)
        buf_outputs_all = buf_labels = buf_logits_all = None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits_all = self.buffer.get_data(
                self.args.minibatch_size,
                transform=self.transform
            )
            buf_outputs_all = self.net.forward_all_heads(buf_inputs)

        total_loss = 0.0
        for head_id, (outputs, optimizer) in enumerate(zip(outputs_all, self.head_optimizers)):
            # last head is the evaluated one -> full weight, earlier heads scaled
            weight = 1.0 if head_id == n_heads - 1 else self.args.local_loss_weight

            head_loss = weight * self.loss(outputs, labels)

            if buf_outputs_all is not None:
                buf_outputs = buf_outputs_all[head_id]
                old_logits = buf_logits_all[:, head_id, :]
                replay_loss = (
                    self.args.alpha * F.mse_loss(buf_outputs, old_logits) +
                    self.args.beta * self.loss(buf_outputs, buf_labels)
                )
                head_loss = head_loss + weight * replay_loss

            # graphs are disjoint per head, so backward/step stay inside the loop
            optimizer.zero_grad()
            head_loss.backward()
            optimizer.step()
            total_loss += head_loss.item()

        with torch.no_grad():
            logits_to_store = torch.stack(
                self.net.forward_all_heads(not_aug_inputs),
                dim=1
            )

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels,
            logits=logits_to_store.data
        )

        return total_loss
