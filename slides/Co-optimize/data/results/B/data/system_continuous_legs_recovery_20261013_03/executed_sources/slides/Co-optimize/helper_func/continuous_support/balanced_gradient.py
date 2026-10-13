"""Compare a bounded set of gradient scales before committing a small step.

First-improvement by scale can perpetually choose tiny translation gains and
never visit a larger direction descent. Rank the inexpensive GRADIENT lines
together; contact-event sampling remains a later fallback, not a derivative.
"""
from .stable_gradient import StableGradientModel, StableGradientSearch


class BalancedGradientSearch(StableGradientSearch):
    @staticmethod
    def proposal_groups(rows, fine=False):
        gradients = [row for row in rows if 'gradient' in row[0]]
        samples = [row for row in rows if 'gradient' not in row[0]]
        groups = [('gradient_multiscale', gradients)] if gradients else []
        # Preserve lazy sampling; at most two complete-load finalists in each
        # visited pool. The modest gradient pool compares gains across tools.
        for label, pool in [('coherent_contact_samples', [r for r in samples if 'coherent' in r[0]]),
                            ('individual_contact_samples', [r for r in samples if 'coherent' not in r[0]])]:
            for start in range(0, len(pool), 16):
                groups.append((label, pool[start:start+16]))
        return groups
