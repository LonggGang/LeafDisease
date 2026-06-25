import torch
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(dataset) -> WeightedRandomSampler:
    """can bang so luong anh giua cac class de train cho deu"""
    # 1. dem so anh cua moi class
    class_counts = [0] * len(dataset.classes)
    for sample in dataset.samples:
        class_counts[sample["label_idx"]] += 1
        
    # 2. tinh trong so cho moi class
    class_weights = [1.0 / (count + 1e-6) for count in class_counts]
    
    # 3. gan trong so cho tung anh
    sample_weights = [class_weights[sample["label_idx"]] for sample in dataset.samples]
    
    # 4. tao sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler
