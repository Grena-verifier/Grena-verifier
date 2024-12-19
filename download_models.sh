#!/bin/bash
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
mkdir -p "$script_dir/models"


# MNIST models
mkdir -p "$script_dir/models/mnist"
cd "$script_dir/models/mnist"

[ ! -s "mnist-net_256x6.onnx" ] && (rm -f mnist-net_256x6.onnx; wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/mnist_fc/onnx/mnist-net_256x6.onnx.gz | gunzip > mnist-net_256x6.onnx || rm -f mnist-net_256x6.onnx)
[ ! -s "convSmallRELU__Point.onnx" ] && (rm -f convSmallRELU__Point.onnx; wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convSmallRELU__Point.onnx)
[ ! -s "convMedGRELU__Point.onnx" ] && (rm -f convMedGRELU__Point.onnx; wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convMedGRELU__Point.onnx)
[ ! -s "convBigRELU__DiffAI.onnx" ] && (rm -f convBigRELU__DiffAI.onnx; wget https://files.sri.inf.ethz.ch/eran/nets/onnx/mnist/convBigRELU__DiffAI.onnx)


# CIFAR10 CNN models
mkdir -p "$script_dir/models/cifar10"
cd "$script_dir/models/cifar10"

[ ! -s "convMedGRELU__PGDK_w_0.0078.onnx" ] && (rm -f convMedGRELU__PGDK_w_0.0078.onnx; wget https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convMedGRELU__PGDK_w_0.0078.onnx)
[ ! -s "convBigRELU__DiffAI.onnx" ] && (rm -f convBigRELU__DiffAI.onnx; wget https://files.sri.inf.ethz.ch/eran/nets/onnx/cifar/convBigRELU__DiffAI.onnx)


# CIFAR10 ResNet models
[ ! -s "resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx" ] && (rm -f resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx; wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/sri_resnet_a/onnx/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx.gz | gunzip > resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx || rm -f resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx)
[ ! -s "resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx" ] && (rm -f resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx; wget -O - https://github.com/ChristopherBrix/vnncomp2022_benchmarks/raw/main/benchmarks/sri_resnet_b/onnx/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx.gz | gunzip > resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx || rm -f resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx)
[ ! -s "resnet_4b.onnx" ] && (rm -f resnet_4b.onnx; wget -O resnet_4b.onnx https://raw.githubusercontent.com/stanleybak/vnncomp2021/main/benchmarks/cifar10_resnet/onnx/resnet_4b.onnx)


# Verify all downloads
cd "$script_dir/models"

# Print header
printf '%80s\n' | tr ' ' '='

check_file() {
    if [ ! -f "$1" ] || [ ! -s "$1" ]; then
        return 1
    fi
    return 0
}

# Track failed files
failed_files=()

# Check MNIST models
for file in mnist/mnist-net_256x6.onnx mnist/convSmallRELU__Point.onnx mnist/convMedGRELU__Point.onnx mnist/convBigRELU__DiffAI.onnx; do
    check_file "$file" || failed_files+=("$file")
done

# Check CIFAR10 CNN models
for file in cifar10/convMedGRELU__PGDK_w_0.0078.onnx cifar10/convBigRELU__DiffAI.onnx; do
    check_file "$file" || failed_files+=("$file")
done

# Check CIFAR10 ResNet models
for file in cifar10/resnet_3b2_bn_mixup_adv_4.0_bs128_lr-1.onnx cifar10/resnet_3b2_bn_mixup_ssadv_4.0_bs128_lr-1_v2.onnx cifar10/resnet_4b.onnx; do
    check_file "$file" || failed_files+=("$file")
done

# Print results
if [ ${#failed_files[@]} -eq 0 ]; then
    echo "All models downloaded SUCCESSFULLY"
else
    for file in "${failed_files[@]}"; do
        echo "FAILED to download \"$file\""
    done
    echo "Try re-running the script to download the missing models."
fi

# Print footer
printf '%80s\n' | tr ' ' '='

exit ${#failed_files[@]}
