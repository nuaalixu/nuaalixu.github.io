---
layout: post
category: Pytorch教程
---

# Dataset, Sampler, DataLoader
## 自上而下的关系
DataLoader：生产mini-batch，供训练；

Sampler：提供一个或多个样本索引，供采样；

Dataset：生产一个或多个样本；

## Dataset
数据集，将数据读取到内存，并提供访问接口。

pytorch实现了两种类型的数据集。

#### map-style
实现了`__getitem()__()` 和`__len__()` 接口，后者通常要被Sampler和DataLoader需要。

dataset\[idx/key\]获取数据，可以通过整数索引，也可以通过关键字索引；

此外，为了加速组batch，可以实现`__getitems__()`，接受索引列表，一次返回多个样本。

```python
class Dataset(Generic[T_co]):
   def __getitem__(self, index) -> T_co:
       raise NotImplementedError
   # def __getitems__(self, indices: List) -> List[T_co]:

   def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
       return ConcatDataset([self, other])

   # No `def __len__(self)` default?
   # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
   # in pytorch/torch/utils/data/sampler.py
```
#### Iterable-style
实现`__iter__()`协议，iter(dataset) 获取读取数据的数据流。
该类型的数据集适合随机读取数据消耗大或不可行的场景，例如读取数据库、远程服务器和实时生成的日志等。

```python
class IterableDataset(Dataset[T_co]):
   def __iter__(self) -> Iterator[T_co]:
       raise NotImplementedError

   def __add__(self, other: Dataset[T_co]):
       return ChainDataset([self, other])

   # No `def __len__(self)` default? Subclasses raise `TypeError` when needed.
   # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
```
使用`DataLoader`的多进程模式读取数据时，`IterableDataset` 在每个子进程创建的迭代器会造成数据重复，需要使用在各子进程进行不同的配置，配置可以实现在dataset的`__iter__()` 方法,或者单独实现然后传递给`Dataloader` 的`worker_init_fn` 参数。

例如，一个可迭代数据顺序返回\[start, end\]的整数，可以通过`range(start, end)`实现，但要支持多进程模式，就得每个worker设置不同的起止点。如下：

```python
>>> class MyIterableDataset(torch.utils.data.IterableDataset):
...     def __init__(self, start, end):
...         super(MyIterableDataset).__init__()
...         assert end > start, "this example code only works with end >= start"
...         self.start = start
...         self.end = end
...
...     def __iter__(self):
...         worker_info = torch.utils.data.get_worker_info()
...         if worker_info is None:  # single-process data loading, return the full iterator
...             iter_start = self.start
...             iter_end = self.end
...         else:  # in a worker process
...             # split workload
...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
...             worker_id = worker_info.id
...             iter_start = self.start + worker_id * per_worker
...             iter_end = min(iter_start + per_worker, self.end)
...         return iter(range(iter_start, iter_end))
...
>>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
>>> ds = MyIterableDataset(start=3, end=7)

>>> # Single-process loading
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
[tensor([3]), tensor([4]), tensor([5]), tensor([6])]

>>> # Mult-process loading with two worker processes
>>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
[tensor([3]), tensor([5]), tensor([4]), tensor([6])]

>>> # With even more workers
>>> print(list(torch.utils.data.DataLoader(ds, num_workers=12)))
[tensor([3]), tensor([5]), tensor([4]), tensor([6])]
```
## Sampler
采样器，通过生成不同的索引或索引列表（batch），来控制从数据集取数据的方式，比如顺序采样器`SequentialSampler`, 随机采样器`RandomSampler` 等。

用于map-style的dataset，默认生成整数索引，故针对索引非整数的dataset，需要定制sampler。

对于iterable-style的dataset，数据读取取决于dataset的`__iter__()`方法，无需利用sampler进行数据读取。

#### *CLASS torch.utils.data.Sampler(data\_source=None)*
* **data\_source** (Dataset) – dataset to sample from. 但是不强制使用。

Sampler的子类要求实现`__iter__()` 和`__len__()`方法，虽然后者不严格要求，但通常被DataLoader 用到。

```python
class Sampler(Generic[T_co]):

   def __init__(self, data_source: Optional[Sized]) -> None:
       pass

   def __iter__(self) -> Iterator[T_co]:
       raise NotImplementedError
```
假设我们要定义一个采样器从短到长地提供字符串数据集的数据（索引），如下所示：

```python
>>> class AccedingSequenceLengthSampler(Sampler[int]):
>>>     def __init__(self, data: List[str]) -> None:
>>>         self.data = data
>>>
>>>     def __len__(self) -> int:
>>>         return len(self.data)
>>>
>>>     def __iter__(self) -> Iterator[int]:
>>>         sizes = torch.tensor([len(x) for x in self.data])
>>>         yield from torch.argsort(sizes).tolist()
```
其中，通过`len()`获取数据集中字符串的长度，然后通过`torch.argsort` 从短到长排序，通过`yield from`一个接一个输出排序后的索引。

#### BatchSampler
如果我们需要采样器一次返回多个索引，即需要BatchSampler，如下所示：

```python
>>> class AccedingSequenceLengthBatchSampler(Sampler[List[int]]):
>>>     def __init__(self, data: List[str], batch_size: int) -> None:
>>>         self.data = data
>>>         self.batch_size = batch_size
>>>
>>>     def __len__(self) -> int:
>>>         return (len(self.data) + self.batch_size - 1) // self.batch_size
>>>
>>>     def __iter__(self) -> Iterator[List[int]]:
>>>         sizes = torch.tensor([len(x) for x in self.data])
>>>         for batch in torch.chunk(torch.argsort(sizes), len(self)):
>>>             yield batch.tolist()
```
通过`torch.chunk()`划分多个索引，然后一次返回一个batch的索引。

#### DistributedSampler
将一个dataset根据rank拆分成不同的子集。方便和`torch.nn.parallel.DistributedDataParallel` 配合使用。

以下是pytorch内置的分布式单样本采样器的实现。

```python
class DistributedSampler(Sampler[_T_co]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        ...

    def __iter__(self) -> Iterator[_T_co]:
        ...
        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
```
简单地通过rank和num\_replicas（默认值等于world\_size）来切分dataset，得到子数据集的索引。

## DataLoader
将`Dataset`的data通过`Sampler`控制的采样方式，组成batch。

DataLoader实例自身是`Iterable`对象，每次迭代生成一个batch的数据，这个batch的形式可以是列表、字典、tensor，和“被batch“的数据有关。

dataset实例必须传入，sampler实例可以传入，也可以通过`batch_size` 、`shuffle`等参数设置后，dataloader会自动生成sampler实例。

#### *CLASS torch.utils.data.DataLoader*
* **dataset** (*Dataset*) – 对map style和iterable style的数据集有不同处理；
* **batch\_size** (*int\**,\* *optional*) – 默认为1，当不为`None`时，data loader产生一个batch的样本。
* **shuffle** (*bool\**,\* *optional*) – set to `True` to have the data reshuffled at every epoch (default: `False`).
* **sampler** (*Sampler* *or* *Iterable\_*,\_ *optional*) – defines the strategy to draw samples from the dataset. Can be any `Iterable` with `__len__` implemented. If specified, `shuffle` must not be specified.
* **batch\_sampler** (*Sampler* *or* *Iterable\_*,\_ *optional*) – like `sampler`, but returns a batch of indices at a time. Mutually exclusive with `batch_size`, `shuffle`, `sampler`, and `drop_last`.
* **num\_workers** (*int\**,\* *optional*) – how many subprocesses to use for data loading. `0` means that the data will be loaded in the main process. (default: `0`)
Each worker collects its loaded batches in a queue and the `DataLoader` will return the next batch from it.
* **collate\_fn** (*Callable\_*,\_ *optional*) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
* **pin\_memory** (*bool\**,\* *optional*) – If `True`, the data loader will copy Tensors into device/CUDA pinned memory before returning them. If your data elements are a custom type, or your `collate_fn` returns a batch that is a custom type, see the example below.
* ...

data loader和其他模块的关系，可见以下伪代码：

```python
class DataLoader(object): 
    ... 

    def __next__(self): 
        if self.num_workers == 0:   
            indices = next(self.sample_iter)  
            batch = self.collate_fn([self.dataset[i] for i in indices]) # this line 
            if self.pin_memory: 
                batch = _utils.pin_memory.pin_memory_batch(batch) 
            return batch
```
`collate_fn`方法是组batch的接口，是将输入类型转换成输出类型的映射。

`dataset` 是一个数据集，此处以map-style为例。

`sample_iter`是batch sampler，返还每个batch size大小的索引列表。

#### 加载批量/非批量数据
`DataLoader` 支持自动将单独的样本整理成batch，利用了`collate_fn` 的默认实现。

当`batch_size`不为`None` 时，data loader产生batch数据，而不是单独的样本。特别的，当dataset是map-style时，也可以通过指定`batch_sampler` ，一次产生多个键，来开启批量数据加载。

批量数据加载，不同数据集内部调用有区别，伪代码如下。

当map-style时：

```python
# 通过索引操作
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```
当iterable-style时：

```python
# 通过next操作
dataset_iter = iter(dataset)
for indices in batch_sampler:
    yield collate_fn([next(dataset_iter) for _ in indices])
```
有时希望关闭自动批量加载，而采用手动打batch或单样本加载。通过设置`batch_size` 和 `batch_sampler` 均为 `None` 来关闭自动打batch。

当不自动打batch时，默认的 `collat​​e_fn` 只是将 NumPy 数组转换为 PyTorch 张量，并保持其他所有内容不变。此时，就需要dataset自身完成组batch的功能。

针对map-style数据集，相当于：

```python
for index in sampler:
    yield collate_fn(dataset[index])
```
针对iterable-style：

```python
for data in iter(dataset):
    yield collate_fn(data)
```
#### collate\_fn
启用或禁用自动批处理时，`collat​​e_fn` 的使用略有不同。

* 关闭：对每个单独的数据样本调用 collat​​e\_fn，并从data loader迭代器产生输出。默认的 `collat​​e_fn` 只是将 NumPy 数组转换为 PyTorch 张量。
* 打开：每次调用 `collat​​e_fn` 时都会使用多个数据样本的列表，此时默认的`collate_fn`将多个样本整理成batch，以便从data loader迭代器中生成。

打开自动批处理时，默认的`collate_fn`行为如下：

* 总是在前面添加一个新维度作为批次维度。
* 自动将 NumPy 数组和 Python 数值转换为 PyTorch 张量。
* 总是保留数据结构：例如每一个样本都是字典，最终输出为保留字典结构，拥有同样的键，但是每个键对应的值是打batch之后的Tensor（如果值不能被转为Tensor，则是列表形式）。

例如，假设batch size是四：

```Plain Text
单样本：0，1，2，3 ...
批处理：tensor([0, 1, 2, 3])
单样本: 'A', 'B', 'C', 'D' ...
批处理：['A', 'B', 'C', 'D']
单样本：{'name': 'A', 'index': 1}
批处理：{'name': ['A', 'B', 'C', 'D'], 'index': tensor([0, 1, 2, 3])}
```
`collate_fn` 可以被用户重载，以定制不同的打batch操作，如不沿第一个维度组batch，padding不同长度的样本，定制数据类型转换为Tensor等。

## 其他工具
#### get\_worker\_info()
`Dataloader` 开启多进程模式时，在某个进程调用该函数，可以获取当前进程`Dataloader`的相关信息。

返回的对象有如下属性，通过`.`访问。

* `id`: the current worker id.
* `num_workers`: the total number of workers.
* `seed`: the random seed set for the current worker. 
* `dataset`: the copy of the dataset object in **this** process.

#### random\_split()
当需要将dataset划分为不同的子集时，例如，划分为训练集和验证集，可以通过`torch.utils.data.random_split`进行。它会根据长度列表，将原始数据集，按照每个长度划分为对应的子集。