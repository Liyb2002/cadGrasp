"""New large sets; no changes to the saved object-task selection files."""
CASES = {
    'seven_chain': list(range(1, 8)),
    'seven_hard': [1, 2, 4, 5, 6, 7, 11],
    'seven_spread': [1, 4, 7, 12, 21, 23, 27],
    'eight_chain': list(range(1, 9)),
    'pose1-10': list(range(1, 11)),
    'eight_hard': [1, 2, 4, 5, 6, 7, 10, 11],
    'eight_spread': [1, 3, 7, 10, 14, 18, 23, 27],
    'nine_chain': list(range(1, 10)),
    'nine_spread': [1, 4, 7, 9, 12, 16, 21, 23, 27],
    'ten_spread': [1, 3, 5, 7, 9, 12, 16, 21, 23, 27],
}

POSE_SET_NAMES = {case:'pose'+'+'.join(map(str,numbers)) for case,numbers in CASES.items()}


def case_directory(root, case):
    """Explicit pose IDs for new output; retain access to historical aliases."""
    named = root/POSE_SET_NAMES[case]
    return named if named.exists() else root/case
