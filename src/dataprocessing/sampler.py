import torch
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Computes class weights based on frequency and returns a WeightedRandomSampler.
    This dynamically handles class imbalance during batch collation, removing the
    need to artificially pad or duplicate the underlying dataset.
    
    Args:
        dataset: An instance of PlantDiseaseDataset (must have .samples and .classes)
        
    Returns:
        WeightedRandomSampler configured to balance minority and majority classes.
    """
    # 1. Count frequencies of each class
    class_counts = [0] * len(dataset.classes)
    for sample in dataset.samples:
        class_counts[sample["label_idx"]] += 1
        
    # 2. Compute weights for each class (inverse frequency)
    # Adding a small epsilon to prevent division by zero in empty classes
    class_weights = [1.0 / (count + 1e-6) for count in class_counts]
    
    # 3. Assign a weight to every single sample in the dataset based on its class
    sample_weights = [class_weights[sample["label_idx"]] for sample in dataset.samples]
    
    # 4. Create the sampler
    # num_samples determines how many iterations constitute one 'epoch'
    # By default, we use len(dataset), but it can be set to (N_TRAIN * len(classes))
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler
