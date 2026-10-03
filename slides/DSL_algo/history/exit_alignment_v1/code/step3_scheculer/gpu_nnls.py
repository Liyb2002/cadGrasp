"""Batched Lawson-Hanson NNLS with seven-dimensional active-set solves."""
def solve(torch, rays, rhs, max_iterations=192):
    # rays[B,M,R], rhs[1,N,R]. Each target has its own active set.
    batch, columns, rows = rays.shape
    targets = rhs.shape[1]
    wt = rays.transpose(1, 2)
    x = torch.zeros((batch, targets, columns), dtype=rays.dtype, device=rays.device)
    active = torch.zeros_like(x, dtype=torch.bool)
    tolerance = 1e-10
    for _ in range(max_iterations):
        gradient = (rhs-x@rays)@wt
        gradient = gradient.masked_fill(active, -torch.inf)
        maximum, column = gradient.max(dim=2)
        moving = maximum > tolerance
        if not moving.any().item():
            break
        active.scatter_(2, column[..., None], active.gather(2, column[..., None]) | moving[..., None])
        for _ in range(columns+1):
            # Small row-space pseudoinverses avoid padded singular M x M systems.
            gram = torch.einsum('bnm,bmi,bmj->bnij', active.to(rays.dtype), rays, rays)
            # cuSOLVER's eigensolver workspace grows strongly with batch size.
            # Bound its batch independently of the number of DSL candidates.
            flat = gram.reshape(-1, rows, rows)
            parts = [torch.linalg.eigh(part) for part in flat.split(256)]
            eigenvalues = torch.cat([v[0] for v in parts]).reshape(batch, targets, rows)
            vectors = torch.cat([v[1] for v in parts]).reshape(batch, targets, rows, rows)
            threshold = eigenvalues[..., -1:]*1e-13
            inverse = torch.where(eigenvalues > threshold, 1/eigenvalues.clamp_min(1e-30), 0.)
            pinv = (vectors*inverse[..., None, :])@vectors.transpose(-1, -2)
            dual = (rhs[:, :, None, :]@pinv).squeeze(2)
            z = (dual@wt)*active
            bad = active & (z <= tolerance)
            repairing = bad.any(dim=2) & moving
            if not repairing.any().item():
                x = torch.where(moving[..., None], z.clamp_min(0.), x)
                break
            ratio = torch.where(bad, x/(x-z).clamp_min(1e-30), torch.inf)
            alpha = ratio.min(dim=2).values.clamp(0., 1.)
            alpha = torch.where(repairing, alpha, 1.)
            updated = x+alpha[..., None]*(z-x)
            x = torch.where(moving[..., None], updated.clamp_min(0.), x)
            active &= (~moving[..., None]) | (x > tolerance)
        else:
            raise RuntimeError('CUDA NNLS inner active-set iteration limit')
    return x
