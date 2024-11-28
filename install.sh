#!/bin/bash
set -e

print_usage() {
    echo "Usage: sudo ./$0 [OPTIONS]"
    echo "Options:"
    echo "  -c, --use-cuda      Whether to install with CUDA support"
    echo "  -h, --help          Display this help message"
    exit 1
}

has_cuda=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--use-cuda)
            has_cuda=1
            shift
            ;;
        -h|--help)
            print_usage
            ;;
        *)
            echo "Error: Unknown parameter $1" >&2
            print_usage
            ;;
    esac
done

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

bash install_libraries.sh

# Install ELINA found at the `./ELINA` dir
cd ELINA
if test "$has_cuda" -eq 1; then
    ./configure -use-cuda -use-deeppoly -use-gurobi -use-fconv
    cd ./gpupoly/
    cmake .
    cd ..
else
    ./configure -use-deeppoly -use-gurobi -use-fconv
fi
make
make install
cd ..
ldconfig
