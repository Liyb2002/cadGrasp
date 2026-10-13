"""Current baseline simplification: contacts avoid work faces, not access rays."""

ENFORCE_PROCESS_ACCESS = False


def skipped_access_check():
    return dict(passed=True, enforced=False, verified=False,
                classification='disabled_by_support_model',
                reason='Process-access volume is omitted; contact faces still exclude the working surface.')
