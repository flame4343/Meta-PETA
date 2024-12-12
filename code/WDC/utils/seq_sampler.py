from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import math
import torch

class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(dataset) / batch_size / num_replicas)) * batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        return iter(indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples])

    def __len__(self):
        return self.num_samples

def distributed_concat(tensor, total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    cat = torch.cat(output_tensors, dim=0)
    return cat[:total_examples]
