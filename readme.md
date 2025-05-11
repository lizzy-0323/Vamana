# Vamana

Vamana is a graph-based index for approximate nearest neighbor (ANN) search, first introduced in [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://suhasjs.github.io/files/diskann_neurips19.pdf).
This repo provides an implementation of Vamana in C++, with go binding for easy usage.

There are two ways for you to use Vamana:

1. Use C++ library in `Vamana.h`
2. Use go binding in `go_api/vamana.go`

## Pre-requisite

- CMake
- OpenMP

## Build

```bash
# To make C++ library
make cc-build
```

## How to use

### Test C++ library

```bash
make cc-build && ./build/main
```

### Test go binding

```bash
go run main.go
```

## Project Structure

```bash
.
├── c_api
│   └── vamana_c_api.cpp # c binding
├── go_api
│   └── vamana_go_api.go # go binding
├── index
│   └── vamana.h # vamana lib
├── main.cpp  # c++ test
├── main.go  # go test
├── README.md
└── CMakeLists.txt
```

## Result

### Random dataset

Using the following parameters, I got **average recall: 90.10%**:

```c++
const uint32_t dimension = 128;    // vector dimension
const uint32_t num_points = 10000; // number of points
const uint32_t R = 128; // maximum degree, larger to obtain better connectivity, R in paper
const uint32_t L = 100; // candidate list size in building, L in paper
const float alpha = 1.2f;       // robust prune parameter, larger to obtain better approximation
const uint32_t k = 10;          // top-k
const uint32_t ef_search = 400; // candidate list size in search
```

## Reference

- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node](https://suhasjs.github.io/files/diskann_neurips19.pdf)

## TODO

- [ ]  inner product
- [ ]  PQ Index
- [ ]  Cache for index
- [ ]  parallel search
