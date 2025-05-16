package vamana

/*
#cgo CFLAGS: -I${SRCDIR}/../c_api
#cgo LDFLAGS: -L${SRCDIR}/../build -lvamana -Wl,-rpath,${SRCDIR}/../build
#include "vamana_c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// VamanaIndex represents a Vamana index
type VamanaIndex struct {
	index *C.VamanaIndex
}

// NewVamanaIndex creates a new Vamana index
func NewVamanaIndex(dimension, maxPoints uint32, alpha float32, R, L, efSearch uint32) (*VamanaIndex, error) {
	index := C.vamana_create_index(C.uint32_t(dimension), C.uint32_t(maxPoints),
		C.float(alpha), C.uint32_t(R), C.uint32_t(L), C.uint32_t(efSearch))
	if index == nil {
		return nil, fmt.Errorf("failed to create vamana index")
	}

	v := &VamanaIndex{index: index}
	runtime.SetFinalizer(v, (*VamanaIndex).Free)
	return v, nil
}

// AddPoint adds a point to the index
func (v *VamanaIndex) AddPoint(point []float32, id uint32) error {
	if len(point) == 0 {
		return fmt.Errorf("point cannot be empty")
	}

	ret := C.vamana_add_point(v.index, (*C.float)(&point[0]), C.uint32_t(id))
	if ret != 0 {
		return fmt.Errorf("failed to add point")
	}
	return nil
}

// BuildIndex builds the index
func (v *VamanaIndex) BuildIndex() error {
	ret := C.vamana_build_index(v.index)
	if ret != 0 {
		return fmt.Errorf("failed to build index")
	}
	return nil
}

// SearchWithStartPoint performs a search from the start point
func (v *VamanaIndex) SearchWithStartPoint(query []float32, startPoint []float32, k uint32) ([]uint32, []float32, error) {
	if v.index == nil {
		return nil, nil, fmt.Errorf("index not initialized")
	}
	if len(query) == 0 || len(startPoint) == 0 {
		return nil, nil, fmt.Errorf("query and start point cannot be empty")
	}

	labels := make([]C.uint32_t, k)
	distances := make([]C.float, k)

	ret := C.vamana_search_with_start_point(v.index, (*C.float)(&query[0]), (*C.float)(&startPoint[0]),
		C.uint32_t(k), (*C.uint32_t)(&labels[0]), (*C.float)(&distances[0]))

	if ret < 0 {
		return nil, nil, fmt.Errorf("search with start point failed")
	}

	// Convert C arrays to Go slices
	resultLabels := make([]uint32, ret)
	resultDistances := make([]float32, ret)
	for i := 0; i < int(ret); i++ {
		resultLabels[i] = uint32(labels[i])
		resultDistances[i] = float32(distances[i])
	}

	return resultLabels, resultDistances, nil
}

// Search performs a k-nearest neighbor search
func (v *VamanaIndex) Search(query []float32, k uint32) ([]uint32, []float32, error) {
	if v.index == nil {
		return nil, nil, fmt.Errorf("index not initialized")
	}
	if len(query) == 0 {
		return nil, nil, fmt.Errorf("query cannot be empty")
	}

	labels := make([]C.uint32_t, k)
	distances := make([]C.float, k)

	ret := C.vamana_search(v.index, (*C.float)(&query[0]), C.uint32_t(k),
		(*C.uint32_t)(&labels[0]), (*C.float)(&distances[0]))

	if ret < 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	// Convert C arrays to Go slices
	result_labels := make([]uint32, ret)
	result_distances := make([]float32, ret)
	for i := 0; i < int(ret); i++ {
		result_labels[i] = uint32(labels[i])
		result_distances[i] = float32(distances[i])
	}

	return result_labels, result_distances, nil
}

// GetPoint retrieves a point from the index by its ID
func (v *VamanaIndex) GetPoint(id uint32, point []float32) error {
	if len(point) == 0 {
		return fmt.Errorf("point buffer cannot be empty")
	}

	ret := C.vamana_get_point(v.index, C.uint32_t(id), (*C.float)(&point[0]))
	if ret != 0 {
		return fmt.Errorf("failed to get point")
	}
	return nil
}

// Save Index
func (v *VamanaIndex) SaveIndex(path string) error {
	ret := C.vamana_save_index(v.index, C.CString(path))
	if ret != 0 {
		return fmt.Errorf("failed to save index")
	}
	return nil
}

// Load Index
func LoadIndex(path string) (*VamanaIndex, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))
	index := C.vamana_load_index(cPath)
	if index == nil {
		return nil, fmt.Errorf("failed to load index")
	}
	v := &VamanaIndex{index: index}
	runtime.SetFinalizer(v, (*VamanaIndex).Free)
	return v, nil
}

// Close frees the index memory
func (v *VamanaIndex) Free() {
	if v.index != nil {
		C.vamana_free_index(v.index)
		v.index = nil
	}
}

func (v *VamanaIndex) GetDataSize() uint32 {
	return uint32(C.vamana_get_data_size(v.index))
}

func (v *VamanaIndex) GetAvgHops() float32 {
	// TODO: implement
	return 0.0
}

func (v *VamanaIndex) GetAvgDistComputations() float32 {
	// TODO: implement
	return 0.0
}

// PrintParams prints the parameters of the Vamana index
func (v *VamanaIndex) PrintParams() {
	if v.index == nil {
		fmt.Println("Index not initialized")
		return
	}
	C.vamana_print_params(v.index)
}
