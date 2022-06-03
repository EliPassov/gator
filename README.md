# README #

GATOR is a deep neural network global channel pruning method utilizing channel hard gating to find which channels to prune based on user defined criteria. 
Criteria is defined via individual weights on each layer, which can represent optimization on FLOPS, Memory or latency, enabling customizing pruning for specific hardware.
While the code is designed for pruning image classification networks, it can be adapted to prune other types of networks.

For more information please refer to the article: 
[Gator: Customizable Channel Pruning of Neural Networks with Gating](https://arxiv.org/abs/2205.15404)

## Supported Architectures ##

Current code supports Sequential networks (E.g. VGG) and ResNet style networks. 
For other types of networks, implementation of the interface GatesModulesMapper, which maps the network's structure for computation of channel pruning costs, is required. 

Integration of frameworks which scan the network architecture (e.g. ONNX) might be implemented in the future. Any contributor willing to take this task is more than welcome and will have the author's support and gratitude.


## Installation ##

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

For training:
- Download the ImageNet dataset from http://www.image-net.org/
  - For arranging the images, consider using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) from [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).
  

## Eval ##

Use main.py to run evaluation, example config:
```shell
python main.py --evaluate --val_data_path [images path] --backup_folder [path] --custom_model CustomResNet50 --batch-size 32 --resume [net weights path] --evaluate
```

## Inference (using pruned networks) ##

A pruned network requires a proper class definition. For ResNet, the _CustomResNet_ class is available.

Pruned Resnet 50 weights optimized for FLOPS and for latency are available [here](https://drive.google.com/drive/folders/19q7v8cLAFdRV2K-ezFSzWpjxASi6mtB_?usp=sharing)



## Training ##

main.py is running the pruning training. For the default parameters setup, it is based on the implementation of [pytorch/examples/imagenet](https://github.com/pytorch/examples/tree/master/imagenet).

### Example training configs ###

#### training ####
```shell
python main.py --train_data_path [imagent train path] --val_data_path [imagenet val path] --backup_folder [path] -a resnet50 --net_with_criterion ResNet50_gating --epochs 30 --lr 0.01 --epoch-lr-step 20 --pretrained --multiprocessing-distributed --dist-url tcp://127.0.0.1:23456 --world-size 1 --rank 0 --subdivision 4 --gating_config_path cfg/resnet_50_flops_config.json
```

#### resume training ####
```shell
python main.py --train_data_path [imagent train path] --val_data_path [imagenet val path] --backup_folder [path] --custom_model CustomResNet50 --epochs 130 --lr 0.0001 --epoch-lr-step 125 --resume [model path] --multiprocessing-distributed --dist-url tcp://127.0.0.1:23456 --world-size 1 --rank 0 --subdivision 4
```

### training parameters ###

* For network training configurations, follow the [train instructions](https://github.com/pytorch/examples/tree/master/imagenet#training) to configure all training parameters

### Gator related parameters ###
* subdivision: Enables training larger batches **overcoming GPU RAM limitations** by dividing each batch into sub-samples and accumulating the gradient. 
Yes, this is compatible with multi-GPU training. Note: Similar to simplified multi GPU training, this makes the training identical to one batch in every aspect save for batch normailzation.
* gating_config_path: Gating configuration in .json format, See examples in /cfg. 
* net_with_criterion: should be set to ResNet_gating, defines the wrapper which maps the network and performs the pruning
* custom_model: Optional, a network architecture not defined in pytorch (alternative to --arch configuration option)
Note: This is required for loading and converting any pruned network.
* backup_folder: where to store model backups
* save_interval: how many epochs to skip between model storage

#### Pruning Gating Configuration ####

Details for the gating configuration

* factor_type: type of weights factor applied based on computation of the weights, currently only supporting : flop_factor / memmory_factor. Note that current implementation requires adjusting one of those, e.g. for latency use multipliers to enhance or reduce relevant factors.
* criteria_weight: pruning loss multiplier (weight of the loss compared to main loss)
* gradient_multiplier: global multiplier effecting gradients on pruning gates weights, acts as a learning rate multiplier for the global learning rate
* gate_init_prob: The probability for a gate to pass, recommended to set close to 1, as lower values produces less desirable results in tests (but can be studied further especially in more NAS scenarios rather than pruning)
* static_total_cost: Static divisor of each individual gate weight, by default (if set to none in the config) set to the sum of all flops/memory used based on criteria. It has the same way as the gradient_multiplier, and is used to normalize the total sum to 1. 
* edge_multipliers: custom weights to adjust the default weights given the factor_type

#### Custom Weights Pruning Network ####

As mentioned, for cust om pruning weights, current implementation uses multipliers on top of the flops/memory weights computed for the network to customize them for latency and other optimization considerations. 

For latency pruning we computed the weight of each hyper graph edge (layer in most cases) by timing running the network with the same edge having 1/2 of its channel, and also computing the equivalent flops reduction.
The results of this computation for ResNet50 can be found in  in results/timinig_resnet_50_edges_7_runs_batch_32.csv. From these results the adjustment weights were computed and stored in cfg/resnet_50_flops_rel_timing_config_batch_32.json.

A notebook for timing and computing the weights can be found in analysis/time_nets.ipynb.


## converting pruned network ##

run net_convert.py for removing pruned channels and creating a de-facto pruned network. Note this requires a module class definition of the pruned network. For ResNet we have defined CustomResNet.

```shell
python net_converter.py --net_with_criterion ResNet_gating --net_name CustomResNet50 --gated_weights_path [path to model] --new_weights_path [path to new model]
```


## Who do I talk to? ##

* If you have any questions or comments you are welcome to contact [Eli Passov](mailto:elipassov@gmail.com?subject[GitHub]Gator)
