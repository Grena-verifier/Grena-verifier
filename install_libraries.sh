# Check if all libraries are already present.
# If so, user might have ran this script twice.
if [ -n "$(ldconfig -p | grep /usr/local/lib/libgmp)" ] && \
  [ -n "$(ldconfig -p | grep /usr/local/lib/libmpfr)" ] && \
  [ -n "$(ldconfig -p | grep /usr/local/lib/libcdd)" ] && \
  [ -n "$(ldconfig -p | grep /usr/local/lib/libgurobi)" ]; then
   echo "All libraries (ie. GMP, MPFR, CDDlib, Gurobi) are already found on this machine. Did you accidentally ran this script twice?"
   exit 0
fi


# Install GMP
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz && \
    tar -xvf gmp-6.1.2.tar.xz && \
    cd gmp-6.1.2 && \
    ./configure --enable-cxx && \
    make && \
    make install && \
    cd .. && \
    rm gmp-6.1.2.tar.xz && \
    ldconfig


# Install MPFR
wget https://files.sri.inf.ethz.ch/eran/mpfr/mpfr-4.1.0.tar.xz && \
    tar -xvf mpfr-4.1.0.tar.xz && \
    cd mpfr-4.1.0 && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm mpfr-4.1.0.tar.xz && \
    ldconfig


# Install CDDlib
wget https://github.com/cddlib/cddlib/releases/download/0.94m/cddlib-0.94m.tar.gz && \
    tar zxf cddlib-0.94m.tar.gz && \
    cd cddlib-0.94m && \
    ./configure && \
    make && \
    make install && \
    cd .. && \
    rm cddlib-0.94m.tar.gz && \
    ldconfig


# Install Gurobi
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz && \
    tar -xvf gurobi9.1.2_linux64.tar.gz && \
    cd gurobi912/linux64/src/build && \
    sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile && \
    make && \
    cp libgurobi_c++.a ../../lib/ && \
    cd ../../ && \
    cp lib/libgurobi91.so /usr/local/lib && \
    python3 setup.py install && \
    cd ../../ && \
    rm gurobi9.1.2_linux64.tar.gz && \
    ldconfig

export GUROBI_HOME="$(pwd)/gurobi912/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export CPATH="${CPATH}:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GUROBI_HOME}/lib:/usr/local/lib

if [ ! -L "/usr/local/include/gurobi_c.h" ]; then
   ln -s ${GUROBI_HOME}/include/gurobi_c.h /usr/local/include/gurobi_c.h
fi

if ! grep -q "GUROBI_HOME" ~/.bashrc; then
    echo "Adding Gurobi environment variables to ~/.bashrc"
    echo "export GUROBI_HOME=\"$(pwd)/gurobi912/linux64\"" >> ~/.bashrc
    echo 'export PATH="${PATH}:${GUROBI_HOME}/bin"' >> ~/.bashrc
    echo 'export CPATH="${CPATH}:${GUROBI_HOME}/include"' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${GUROBI_HOME}/lib:/usr/local/lib' >> ~/.bashrc
fi
