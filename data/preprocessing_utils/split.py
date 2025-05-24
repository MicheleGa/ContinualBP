import random

def split_trainvalid_fixed(sample_n, txf_percentage=0.2):
    train_n = int(sample_n * txf_percentage)
    train = list(range(train_n))
    valid = list(range(train_n, sample_n))
    return train, valid


def split_train_val_test(subject_sample_ids, split_ratio, shuffle=False):
    """
    Splits a list of sample IDs into disjoint train, validation, and test sets based on the given split ratios.

    Parameters:
        subject_sample_ids (list): List of sample IDs to be split.
        split_ratio (list): List of three floats representing the desired ratios for train, validation, and test sets.
        shuffle (bool, optional): Whether to shuffle the sample IDs before splitting. Defaults to True.

    Returns:
        tuple: A tuple containing three lists: train_idx, val_idx, and test_idx.
    """

    if sum(split_ratio) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    if shuffle:
        random.shuffle(subject_sample_ids)  # Shuffle the sample IDs to ensure randomness

    total_samples = len(subject_sample_ids)
    train_size = int(total_samples * split_ratio[0])
    val_size = int(total_samples * split_ratio[1])

    train_idx = subject_sample_ids[:train_size]
    val_idx = subject_sample_ids[train_size:train_size + val_size]
    test_idx = subject_sample_ids[train_size + val_size:]

    return train_idx, val_idx, test_idx


def split_train_val_test_personalization(subject_sample_ids, fixed_train_size, fixed_val_size=50, fixed_test_size=70):
    """
    Chronologically splits a list of sample IDs into disjoint personalization, validation, and test sets.
    The personalization set is taken from the beginning. The fixed-size test set is taken from the end
    of the remaining samples, and the fixed-size validation set is taken from the samples immediately
    preceding the test set.

    Parameters:
        subject_sample_ids (list): List of sample IDs to be split.
        fixed_train_size (int): Fixed number of samples for the training set (at the beginning of subject_sample_ids).
        fixed_val_size (int): Fixed number of samples for the validation set (before the test set at the end of remaining).
        fixed_test_size (int): Fixed number of samples for the test set (from the end of remaining).

    Returns:
        tuple: A tuple containing three lists: personalization_idx, val_idx, and test_idx.
    """

    if fixed_train_size > len(subject_sample_ids):
        raise ValueError("Personalization sample number cannot be greater than the total number of samples")
    
    # Take the first personalization_sample_number samples for training
    train_idx = subject_sample_ids[:fixed_train_size]
    
    # The remaining samples
    remaining_samples = subject_sample_ids[fixed_train_size:]
    num_remaining = len(remaining_samples)

    # Take the last fixed_test_size samples for the test set
    test_idx = remaining_samples[num_remaining - fixed_test_size:]

    # Take the fixed_val_size samples before the test set for validation
    val_idx_end = num_remaining - fixed_test_size
    val_idx_start = val_idx_end - fixed_val_size
    val_idx = remaining_samples[val_idx_start:val_idx_end]
    
    return train_idx, val_idx, test_idx