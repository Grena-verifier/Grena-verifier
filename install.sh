#!/bin/bash
set -e

print_usage() {
    echo "Usage: sudo ./$0 [OPTIONS]"
    echo "Options:"
    echo "  -h, --help          Display this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            print_usage
            ;;
        *)  
            echo "Error: Unknown parameter $1" >&2
            print_usage
            ;;
    esac
done

# Check for CUDA support
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "Error: CUDA support not found. Please ensure NVIDIA drivers and CUDA toolkit are properly installed."
    exit 1
fi

# Check if user ran this script with sudo/root permission
if [ "$EUID" -ne 0 ]; then
    echo "WARNING: Not running with sudo/root permission."
    read -p "You may encounter problems installing the dependencies. Continue? (Y/N) " -n 1 -r
    echo    # Move to a new line
    while ! [[ $REPLY =~ ^[YyNn]$ ]]; do
        read -p "Please enter Y or N: " -n 1 -r
        echo
    done

    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Operation cancelled."
        exit 1
    fi
fi

# Install the other necessary libraries
bash install_libraries.sh

# Install ELINA found at the `./ELINA` dir with CUDA support
cd ELINA
./configure -use-cuda -use-deeppoly -use-gurobi -use-fconv
cd ./gpupoly/
cmake .
cd ..
make
make install
cd ..
ldconfig
