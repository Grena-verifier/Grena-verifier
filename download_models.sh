#!/bin/bash
set -e
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

mkdir -p "$script_dir/models"


# MNIST models
mkdir -p "$script_dir/models/mnist"
cd "$script_dir/models/mnist"

[ ! -f "mnist-net_256x6.onnx" ] && wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx.gz | gunzip > mnist-net_256x6.onnx
[ ! -f "convSmallRELU__Point.onnx" ] && wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__Point.onnx
[ ! -f "convMedGRELU__Point.onnx" ] && wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__Point.onnx
[ ! -f "convBigRELU__DiffAI.onnx" ] && wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convBigRELU__DiffAI.onnx


# CIFAR10 CNN models
mkdir -p "$script_dir/models/cifar10"
cd "$script_dir/models/cifar10"

[ ! -f "convMedGRELU__PGDK_w_0.0078.onnx" ] && wget https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.0078.onnx
[ ! -f "convBigRELU__DiffAI.onnx" ] && wget https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convBigRELU__DiffAI.onnx


# CIFAR10 ResNet models
[ ! -f "resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx" ] && wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx.gz | gunzip > resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx
[ ! -f "resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx" ] && wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/sri_resnet_b/onnx/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx.gz | gunzip > resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx
[ ! -f "resnet_4b.onnx" ] && wget -O resnet_4b.onnx https://raw.githubusercontent.com/stanleybak/vnncomp2021/main/benchmarks/cifar10_resnet/onnx/resnet_4b.onnx
