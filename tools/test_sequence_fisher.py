"""Numerical audit against exact KL curvature, including cross-token coupling."""
import itertools
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptq_utils.sequence_fisher import score_losses, SequenceFisherCache


class SequenceFisherTests(unittest.TestCase):
    def test_exact_categorical_kl_hessian(self):
        torch.manual_seed(5)
        y = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        a = torch.randn(2, 3, 4, dtype=torch.float64)
        def logits(z):
            return torch.einsum('tvh,h->tv', a, z.reshape(-1)).tanh()
        probs = logits(y).softmax(-1).detach()
        def kl(z):
            return (probs * (probs.log() - logits(z).log_softmax(-1))).sum() / 2
        hessian = torch.autograd.functional.hessian(kl, y).reshape(4, 4)
        fisher = torch.zeros_like(hessian)
        for labels in itertools.product(range(3), repeat=2):
            nll = -logits(y).log_softmax(-1)[torch.arange(2), list(labels)].sum()
            score = torch.autograd.grad(nll, y)[0].reshape(-1)
            probability = probs[torch.arange(2), list(labels)].prod()
            fisher += probability * torch.outer(score, score) / 2
        torch.testing.assert_close(fisher, hessian, atol=1e-12, rtol=1e-12)
        self.assertGreater(float(fisher[:2, 2:].norm()), 0.01)

    def test_projection_and_gradient_against_dense_matrix(self):
        torch.manual_seed(9)
        delta = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
        scores = torch.randn(2, 4, 3, 2, dtype=torch.float64)
        seq, token = score_losses(delta, scores)
        dense = scores.flatten(2).transpose(1, 2) @ scores.flatten(2) / 4
        d = delta.flatten(1)
        expected = torch.einsum('bi,bij,bj->', d, dense, d) / 12
        torch.testing.assert_close(seq, expected)
        torch.testing.assert_close(torch.autograd.grad(seq, delta, retain_graph=True)[0],
                                   torch.autograd.grad(expected, delta)[0])
        blocks = dense.reshape(2, 3, 2, 3, 2)
        expected_token = sum(torch.einsum('bi,bij,bj->', delta[:, t], blocks[:, t, :, t],
                                         delta[:, t]) for t in range(3)) / 12
        torch.testing.assert_close(token, expected_token)
        self.assertTrue(torch.autograd.gradcheck(lambda d: score_losses(d, scores)[0], (delta,)))
        # A one-token sequence cannot contain inter-token terms.
        single = score_losses(delta[:, :1], scores[:, :, :1])
        torch.testing.assert_close(*single)

    def test_cache_scale_alignment_and_noop(self):
        cache = SequenceFisherCache(2, 1, [0], 'sequence', 0)
        cache.active = True
        grad = torch.arange(12).reshape(2, 3, 2).float() * 1000
        cache.capture(0, grad, 1000)
        cache.validate()
        torch.testing.assert_close(cache.batch(0, [1, 0], 'cpu')[:, 0].float(), grad[[1, 0]] / 1000)
        loss = torch.tensor(2., requires_grad=True)
        self.assertIs(cache.blend(loss, None, 0, None, 'cpu', 'test'), loss)
        with self.assertRaises(RuntimeError):
            cache.capture(0, grad, 1000)

    def test_cross_mode_preserves_aggregate_and_adds_only_interactions(self):
        torch.manual_seed(14)
        delta = torch.randn(2, 3, 2, requires_grad=True)
        scores = torch.randn(2, 2, 3, 2)
        aggregate = torch.tensor(4.25)
        sequence, token = score_losses(delta, scores)
        cache = SequenceFisherCache(2, 2, [0], 'cross', 1.0)
        cache.cpu[0] = scores
        result = cache.blend(aggregate, delta, 0, [0, 1], 'cpu', 'mlp.up_proj')
        torch.testing.assert_close(result, aggregate + sequence - token)
        actual_grad = torch.autograd.grad(result, delta, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(sequence - token, delta)[0]
        torch.testing.assert_close(actual_grad, expected_grad)


if __name__ == '__main__':
    unittest.main()
