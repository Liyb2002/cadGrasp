"""Conservative radius cap retaining a certified common insertion direction."""


def radius_cap(initial,maximum,tolerance,certify):
    if not certify(initial):
        raise ValueError('The actual initial contact has no certified common insertion direction')
    if maximum==initial or certify(maximum):
        return maximum,dict(limited=False,certified_radius_m=maximum,upper_uncertified_radius_m=None,iterations=0)
    low,high=initial,maximum
    iterations=0
    while high-low>tolerance:
        middle=(low+high)/2
        if certify(middle):low=middle
        else:high=middle
        iterations+=1
    return low,dict(limited=True,certified_radius_m=low,upper_uncertified_radius_m=high,
                    iterations=iterations,reason='No certified common insertion direction at the upper bracket',
                    exact_maximum_claimed=False)
