FROM ubuntu:24.04

ARG PETSC_VERSION
ARG CC
ARG CXX

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PETSC_INSTALL_DIR=/opt/petsc/
RUN mkdir -p $PETSC_INSTALL_DIR

# Install dependencies
RUN apt-get update && apt-get install -y \
    gcc-13 g++-13 clang-18 clang++-18 \
    cmake openmpi-bin libopenmpi-dev \
    libmetis-dev libparmetis-dev \
    libeigen3-dev python3-numpy \
    libopenblas-dev liblua5.3-dev \
    libomp-dev libgomp1 wget git jq curl gmsh && \
    rm -rf /var/lib/apt/lists/*

# Install libxsmm (using build-args for compiler)
RUN git clone --depth 1 --branch 1.17 https://github.com/libxsmm/libxsmm.git && \
    cd libxsmm && make generator CC="$CC" CXX="$CXX" -j $(nproc) && \
    cp bin/libxsmm_gemm_generator /usr/bin && cd .. && rm -rf libxsmm

# Get and install PETSc (using build-args for compiler and PETSc version)
RUN echo "Using PETSc version $PETSC_VERSION" && \
    wget https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz && \
    tar -xf petsc-${PETSC_VERSION}.tar.gz && rm -rf petsc-${PETSC_VERSION}.tar.gz && cd petsc-${PETSC_VERSION} && \
    PETSC_DIR=$(pwd) && ./configure --with-fortran-bindings=0 --with-debugging=0 \
    --with-memalign=32 --with-64-bit-indices \
    --with-cc="$CC" --with-cxx="$CXX" --with-fc=0 --prefix=$PETSC_INSTALL_DIR \
    --COPTFLAGS="-g -O3" --CXXOPTFLAGS="-g -O3" --with-mpi-dir=/usr/lib/x86_64-linux-gnu/openmpi && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt -j$(nproc) && \
    make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt install && \
    rm -rf petsc-${PETSC_VERSION}.tar.gz petsc-${PETSC_VERSION}

# Save PETSc version for reference
RUN echo "$PETSC_VERSION" > /opt/petsc/version.txt 
