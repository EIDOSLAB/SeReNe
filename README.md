# SeReNe: Sensitivity based Regularization of Neurons for Structured Sparsity in Neural Networks
[![DOI](https://zenodo.org/badge/doi/10.1109/TNNLS.2021.3084527.svg)](http://dx.doi.org/10.1109/TNNLS.2021.3084527)
[![arXiv](https://img.shields.io/badge/arXiv-2102.03773-b31b1b.svg)](https://arxiv.org/abs/2102.03773)

Please cite this work as

```latex
@ARTICLE{9456024,
  author={Tartaglione, Enzo and Bragagnolo, Andrea and Odierna, Francesco and Fiandrotti, Attilio and Grangetto, Marco},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={SeReNe: Sensitivity-Based Regularization of Neurons for Structured Sparsity in Neural Networks}, 
  year={2021},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TNNLS.2021.3084527}}

```

## Requirements
* PyTorch >= 1.6.0
* CUDA >= 10.1
* Tensorboard

## Running code
```
python main.py \
-model [architecture] \
-dataset [training dataset] \
-sensitivity [sensitivity type] \
-lmbda [lambda value] \
-device [cuda:id or cpu] \
-valid_size [percentage of training dataset to use as validation] \
```

## Implemented architectures
Additionally, we provide some implementation of classical architectures

##### MNIST and Fashion-MNIST
LeNet300 | LeNet5

##### CIFAR10
resnet32 | VGG16

##### CIFAR100
AlexNet

## Adding new models and Datasets

### Models
To prune a model not included in the repository simply add a new line to the `_load` function in `getters.py`,
following the template:
```
if model == "mymodel":
    return MyModel()
```
Then add the same model name in `config.py` in `AVAILABLE_MODELS = [... , "mymodel"]`, when done you will be able
to use your model via the argumen `-model mymodel`.

### Datasets
Currently the available Dataloaders are loaded via ``EIDOSearch/dataloaders/get_dataloader``.
To add a new Dataloader just change the content of ``get_dataloaders`` in `NetworkPruning/src/utilities/getters.py`
making sure to return ``train_loader, valid_loader, test_loader``, i.e. a train Dataloader, a validation Dataloader and a test Dataloader.
To build a function that returns such items you can use the following template, modifying it to your needs:
```
def get_data_loaders(data_dir, batch_size, valid_size, shuffle, num_workers, pin_memory, random_seed=0):
    """
    Build and returns a torch.utils.data.DataLoader for the torchvision.datasets.MNIST DataSet.
    :param data_dir: Location of the DataSet or where it will be downloaded if not existing.
    :param batch_size: DataLoader batch size.
    :param valid_size: If greater than 0 defines a validation DataLoader composed of $valid_size * len(train DataLoader)$ elements.
    :param shuffle: If True, reshuffles data.
    :param num_workers: Number of DataLoader workers.
    :param pin_memory: If True, uses pinned memory for the DataLoader.
    :param random_seed: Value for generating random numbers.
    :return: (train DataLoader, validation DataLoader, test DataLoader) if $valid_size > 0$, else (train DataLoader, test DataLoader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = os.path.join(data_dir, "MNIST")

    train_dataset = datasets.MNIST(
        root=data_dir, train=True,
        download=True, transform=transform,
    )

    if valid_size > 0:
        valid_dataset = datasets.MNIST(
            root=data_dir, train=True,
            download=True, transform=transform,
        )

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    test_dataset = datasets.MNIST(
        root=data_dir, train=False,
        download=True, transform=transform
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader, test_loader) if valid_size > 0 else (train_loader, test_loader)
```

## Other optional arguments
```
optional arguments:
  -h, --help            show this help message and exit
  -epochs EPOCHS        Number of training epochs. Default = 1000.
  -lr LR                PyTorch optimizer's learning rate. Default = 0.1.
  -lmbda LMBDA          Sensitivity lambda. Default = 0.0001.
  -twt TWT              Threshold worsening tolerance. Default = 0.
  -pwe PWE              Plateau waiting epochs. Default = 0.
  -mom MOM              Momentum. Default = 0
  -nesterov             Use Nesterov momentum. Default = False.
  -wd WD                Weight decay. Default = 0.
  -model {lenet300,lenet5,alexnet,vgg16_1,vgg16_2,resnet32,resnet18,resnet101}
                        Neural network architecture.
  -ckp_path CKP_PATH    Path to model state_dict.
  -dataset {mnist,fashion-mnist,cifar10,cifar100,imagenet}
                        Dataset
  -valid_size VALID_SIZE
                        Percentage of training dataset to use as validation.
                        Default = 0.1.
  -data_dir DATA_DIR    Folder containing the dataset. Default = data.
  -train_batch_size TRAIN_BATCH_SIZE
                        Batch size. Default = 100.
  -test_batch_size TEST_BATCH_SIZE
                        Batch size. Default = 100.
  -cross_valid          Perform cross validation. Default = False.
  -sensitivity {,serene}
                        Sensitivty optimizer.
  -serene_type {full,lower-bound,local}
                        For SERENE regularization only. Specify which version.
  -serene_alpha SERENE_ALPHA
                        For SERENE regularization only. Alpha value i.e.
                        1/number of classes in the dataset.
  -lr_cycling           Learning rate cycling. Default = False.
  -cycle_up CYCLE_UP    Number of epochs before increasing the LR.
  -cycle_down CYCLE_DOWN
                        Number of epochs before decreasing the LR.
  -mask_params          Pruned parameters mask. Default = False.
  -mask_neurons         Pruned neurons mask. Default = False.
  -batch_pruning        Activate batch-wise pruning, requires -prune_batches
                        and -test_batches. Default = False.
  -prune_iter PRUNE_ITER
                        Defines how many batch iterations should pass before a
                        pruning stepin dataset fractions e.g. 5 = 1/5 of the
                        dataset.
  -test_iter TEST_ITER  Defines how many batch iterations should pass before
                        testingin dataset fractions e.g. 5 = 1/5 of the
                        dataset.
  -device DEVICE        Device (cpu, cuda:0, cuda:1, ...). Default = cpu.
  -seed SEED            Sets the seed for generating random numbers. Default =
                        0.
  -name NAME            Run name. Default = test.
  -dev                  Development mode. Default = False.
```
