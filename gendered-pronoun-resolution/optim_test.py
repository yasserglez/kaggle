import unittest

import torch
from torch.nn import Parameter

from optim import lr_schedule, Adam


class LRScheduleTest(unittest.TestCase):

    def test_constant(self):
        lr = 1e-2
        for step in range(1, 101):
            for total_steps in (None, 100):
                scheduled_lr = lr_schedule(
                    step, total_steps=total_steps,
                    lr=lr, lr_warmup=None, lr_cooldown=None)
                assert scheduled_lr == lr

    def test_warmup_only(self):
        expected_values = [
            (1, 0.01),
            (5, 0.05),
            (10, 0.1),
            (15, 0.1),
            (20, 0.1),
        ]
        for step, expected_lr in expected_values:
            scheduled_lr = lr_schedule(
                step, total_steps=20,
                lr=0.1, lr_warmup=0.5, lr_cooldown=None)
            self.assertAlmostEqual(scheduled_lr, expected_lr)

    def test_warmup_and_linear_cooldown(self):
        lr = 1e-4
        expected_values = [
            (1, 0.1 * lr),
            (5, 0.5 * lr),
            (10, lr),
            (55, 0.5 * lr),
            (91, 0.1 * lr),
            (100, 0),
        ]
        for step, expected_lr in expected_values:
            scheduled_lr = lr_schedule(
                step, total_steps=100,
                lr=lr, lr_warmup=0.1, lr_cooldown='linear')
            self.assertAlmostEqual(scheduled_lr, expected_lr)


class AdamTest(unittest.TestCase):

    def assertListAlmostEqual(self, list_a, list_b, *args, **kwargs):
        self.assertEqual(len(list_a), len(list_b))
        for a, b in zip(list_a, list_b):
            self.assertAlmostEqual(a, b, *args, **kwargs)

    def test_sphere(self):
        optimizer_kwargs = [
            {'lr': 0.1},
            {'lr': 0.3, 'weight_decay': 1e-4},
            {'lr': 0.1, 'max_grad_norm': 1.0},
            {'lr': 0.2, 'lr_warmup': 0.5},
            {'lr': 0.3, 'lr_warmup': 0.1, 'lr_cooldown': 'linear'},
        ]
        for kwargs in optimizer_kwargs:
            # https://www.sfu.ca/~ssurjano/spheref.html
            xopt = torch.randn(10)  # shift optimum
            x = Parameter(torch.zeros_like(xopt).uniform_(-5.12, 5.12))
            loss_func = torch.nn.MSELoss()

            named_parameters = (('x', x), )
            total_steps = 1000
            optimizer = Adam(named_parameters, total_steps=total_steps, **kwargs)
            for _ in range(total_steps):
                loss = loss_func(x, xopt)
                loss.backward()
                optimizer.step()
                x.grad.detach_()
                x.grad.zero_()
            self.assertListAlmostEqual(x.tolist(), xopt.tolist(), places=2)


if __name__ == '__main__':
    unittest.main()
