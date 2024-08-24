######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2024.04.30
# filenaem: reid_sampler.py
# function: REbuild Sampler for Metric Learning.
# reference: https://github.com/JDAI-CV/fast-reid
######################################################
import copy
import math
import torch
import pickle
import itertools
from collections import defaultdict
from typing import Optional, List

import numpy as np
# from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler as Sampler
import torch.distributed as dist


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    # if len(buffer) > 1024 ** 3:
    #     logger = logging.getLogger(__name__)
    #     logger.warning(
    #         "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
    #             get_rank(), len(buffer) / (1024 ** 3), device
    #         )
    #     )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).
    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reorder_index(batch_indices, world_size):
    r"""Reorder indices of samples to align with DataParallel training.
    In this order, each process will contain all images for one ID, triplet loss
    can be computed within each process, and BatchNorm will get a stable result.
    Args:
        batch_indices: A batched indices generated by sampler
        world_size: number of process
    Returns:

    """
    mini_batchsize = len(batch_indices) // world_size
    # for i in range(0, world_size):
    #     mini_batchdata = batch_indices[i*mini_batchsize:(i+1)*mini_batchsize]
    #     np.random.shuffle(mini_batchdata)
    #     batch_indices[i*mini_batchsize:(i+1)*mini_batchsize] = mini_batchdata
    
    reorder_indices = []
    for i in range(0, mini_batchsize):
        for j in range(0, world_size):
            reorder_indices.append(batch_indices[i + j * mini_batchsize])
    return reorder_indices


def shared_random_seed():
    """
    Returns:
        int: a random number that is the same across all workers.
            If workers need a shared RNG, they can use this shared seed to
            create one.
    All workers must call this function, otherwise it will deadlock.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - labels (list): list of (pid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(
            self, 
            targets: list, 
            ids: list, 
            mini_batch_size: int, 
            num_instances: int, 
            world_size: Optional[int] = None,
            seed: Optional[int] = None, 
            drop_last: bool = True,
        ):
        assert num_instances is not None, "Must specify num_instance"
        self.ids = ids
        self.num_instances = num_instances
        self.num_pids_per_batch = mini_batch_size // self.num_instances

        self._rank = get_rank()
        self.num_replicas = world_size # get_world_size()
        self.batch_size = mini_batch_size * self.num_replicas

        self.id_index = defaultdict(list)
        # self.num_ = len(targets) // self._world_size // self.batch_size * self.batch_size
        if drop_last and len(targets) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(targets) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(targets) / self.num_replicas)

        targets = np.array(targets)
        for id in ids:
            self.id_index.update({id: np.where(targets == id)[0].tolist()})

        if seed is None:
            seed = shared_random_seed()
        self._seed = int(seed)
    
    def __len__(self):
        return self.num_samples 

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self.num_replicas)

    def _infinite_indices(self):
        np.random.seed(self._seed)
        # while True:
        avl_pids = copy.deepcopy(self.ids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avl_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(avl_pids, self.num_pids_per_batch, replace=False).tolist()
            for pid in selected_pids:
                # Register pid in batch_idxs_dict if not
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.id_index[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avl_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avl_idxs.pop(0))

                if len(avl_idxs) < self.num_instances:
                    avl_pids.remove(pid)

            if len(batch_indices) == self.batch_size:
                yield from reorder_index(batch_indices, self.num_replicas)
                batch_indices = []


if __name__ == "__main__":
    labels = np.array(list(range(64))*64)
    sample = NaiveIdentitySampler(labels, list(range(64)), 64, 4, world_size=8, seed=42)
    data = list(zip(*(list(range(4096)), labels)))
    # data = torch.tensor(data)#, dtype=torch.long)
    # from local_lib.data.dataset_factory import create_custom_dataset
    # from timm.data import create_loader
    # dataset_train = create_custom_dataset("reid_data", "dataset/optimize_task3", batch_size=64, is_training=True, transform=None, num_instance=4)
    # print(data)
    # sample = NaiveIdentitySampler(list(range(20)),num_samples=10,replacement=False)
    loader_train = torch.utils.data.DataLoader(data, batch_size=64, sampler=sample)
    # loader_train = create_loader(dataset_train, input_size=(3, 224, 224), batch_size=64, shuffle=False)
    # loader_train.sampler = sample
    for i, d in enumerate(loader_train):
        print(i, d[1])
        if i == 10:
            break