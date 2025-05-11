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
)

// VamanaIndex represents a Vamana index
type VamanaIndex struct {
	index *C.VamanaIndex
}

// NewVamanaIndex creates a new Vamana index
func NewVamanaIndex(dimension, maxPoints uint32, alpha float32, R, L uint32) (*VamanaIndex, error) {
	index := C.vamana_create_index(C.uint32_t(dimension), C.uint32_t(maxPoints),
		C.float(alpha), C.uint32_t(R), C.uint32_t(L))
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

// Search performs a k-nearest neighbor search
func (v *VamanaIndex) Search(query []float32, k, efSearch uint32) ([]uint32, []float32, error) {
	if len(query) == 0 {
		return nil, nil, fmt.Errorf("query cannot be empty")
	}

	resultIds := make([]C.uint32_t, k)
	resultDistances := make([]C.float, k)

	ret := C.vamana_search(v.index, (*C.float)(&query[0]), C.uint32_t(k),
		C.uint32_t(efSearch), &resultIds[0], &resultDistances[0])

	if ret < 0 {
		return nil, nil, fmt.Errorf("search failed")
	}

	// Convert C arrays to Go slices
	ids := make([]uint32, ret)
	distances := make([]float32, ret)
	for i := 0; i < int(ret); i++ {
		ids[i] = uint32(resultIds[i])
		distances[i] = float32(resultDistances[i])
	}

	return ids, distances, nil
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
	index := C.vamana_load_index(C.CString(path))
	if index == nil {
		return nil, fmt.Errorf("failed to load index")
	}
	return &VamanaIndex{index: index}, nil
}

// Close frees the index memory
func (v *VamanaIndex) Free() {
	if v.index != nil {
		C.vamana_free_index(v.index)
		v.index = nil
	}
}
