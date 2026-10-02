"""Split a batch of dishes across cuisines.

How many dishes to look for in each cuisine is a portfolio decision, separate
from how good an individual dish is. The cuisines the menu barely covers get
more of the batch; no dish gets a better score for it.
"""


def allocate_quota(total: int, sku_counts: dict[str, int]) -> dict[str, int]:
    """Split `total` dishes across the cuisines in `sku_counts`, favouring the
    ones with fewer existing SKUs.

    A cuisine's share is proportional to 1 / sqrt(1 + its SKU count): an empty
    cuisine clearly leads, but every cuisine keeps some room to explore. Whole
    dishes are handed out by largest remainder, ties going to the cuisine
    listed first.
    """
    if total <= 0 or not sku_counts:
        return {cuisine: 0 for cuisine in sku_counts}

    weights = {cuisine: (1 + max(count, 0)) ** -0.5 for cuisine, count in sku_counts.items()}
    scale = total / sum(weights.values())
    exact = {cuisine: weight * scale for cuisine, weight in weights.items()}

    quota = {cuisine: int(share) for cuisine, share in exact.items()}
    by_remainder = sorted(exact, key=lambda cuisine: exact[cuisine] - quota[cuisine], reverse=True)
    for cuisine in by_remainder[: total - sum(quota.values())]:
        quota[cuisine] += 1
    return quota
