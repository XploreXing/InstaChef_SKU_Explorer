"""How a batch of dishes is split across cuisines."""
import pytest

from utils.quota import allocate_quota

# SKU counts per cuisine on 2026-10-02, in the order the UI lists them.
SKU_COUNTS = {
    "Mexican": 0, "Korean": 12, "Japanese": 14,
    "Thai": 14, "Chinese": 14, "Singaporean/Malay": 36,
}


def test_empty_cuisine_gets_the_largest_share():
    assert allocate_quota(10, SKU_COUNTS) == {
        "Mexican": 5, "Korean": 1, "Japanese": 1,
        "Thai": 1, "Chinese": 1, "Singaporean/Malay": 1,
    }


def test_larger_batch_keeps_the_same_proportions():
    assert allocate_quota(30, SKU_COUNTS) == {
        "Mexican": 14, "Korean": 4, "Japanese": 4,
        "Thai": 3, "Chinese": 3, "Singaporean/Malay": 2,
    }


@pytest.mark.parametrize("total", [1, 2, 7, 10, 23, 60])
def test_quotas_always_add_up_to_the_total(total):
    assert sum(allocate_quota(total, SKU_COUNTS).values()) == total


def test_single_cuisine_gets_the_whole_batch():
    """With one cuisine selected there is nothing to split."""
    assert allocate_quota(10, {"Singaporean/Malay": 36}) == {"Singaporean/Malay": 10}


def test_equal_counts_split_evenly_and_leftovers_go_to_the_first_listed():
    assert allocate_quota(5, {"Thai": 14, "Chinese": 14, "Japanese": 14}) == {
        "Thai": 2, "Chinese": 2, "Japanese": 1,
    }


@pytest.mark.parametrize("total", [0, -3])
def test_nothing_to_hand_out(total):
    assert allocate_quota(total, {"Thai": 14, "Mexican": 0}) == {"Thai": 0, "Mexican": 0}


def test_no_cuisines():
    assert allocate_quota(10, {}) == {}
