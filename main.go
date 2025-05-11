package main

import (
	"fmt"
	"math/rand"
	"sort"
	vamana "vamana/go_api"
)

// set parameters
const (
	dimension  = 128  // vector dimension
	numPoints  = 1000 // number of points
	numQueries = 100  // number of queries
	R          = 56   // maximum degree (R in paper)
	L          = 100  // length of candidate list (L in paper)
	alpha      = 1.2  // robust prune parameter
	k          = 10   // top-k
	efSearch   = 200  // search-time candidate list length (L in paper)
)

func generateRandomVector(dimension uint32) []float32 {
	vec := make([]float32, dimension)
	for i := range vec {
		vec[i] = rand.Float32()*2 - 1 // [-1, 1]
	}
	return vec
}

func computeDistance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(sum)
}

func TestVamanaBasic() {

	// create index
	index, err := vamana.NewVamanaIndex(dimension, numPoints, alpha, R, L)
	if err != nil {
		fmt.Println("Failed to create index:", err)
		return
	}
	defer index.Free()

	points := make([][]float32, numPoints)
	for i := uint32(0); i < numPoints; i++ {
		points[i] = generateRandomVector(dimension)
		err := index.AddPoint(points[i], i)
		if err != nil {
			fmt.Println("Failed to add point:", err)
			return
		}
	}

	err = index.BuildIndex()
	if err != nil {
		fmt.Println("Failed to build index:", err)
		return
	}
	fmt.Println("Index built success")

	queries := make([][]float32, numQueries)
	for i := uint32(0); i < numQueries; i++ {
		queries[i] = generateRandomVector(dimension)
	}

	// calculate average recall
	totalRecall := 0.0
	for i := uint32(0); i < numQueries; i++ {
		query := queries[i]

		// calculate ground truth
		var exactDistances []struct {
			id       uint32
			distance float32
		}
		for j := uint32(0); j < numPoints; j++ {
			dist := computeDistance(query, points[j])
			exactDistances = append(exactDistances, struct {
				id       uint32
				distance float32
			}{j, dist})
		}

		// Sort ground truth
		sort.Slice(exactDistances, func(i, j int) bool {
			return exactDistances[i].distance < exactDistances[j].distance
		})

		// Search
		ids, distances, err := index.Search(query, k, efSearch)
		if err != nil {
			fmt.Printf("Search failed for query %d: %v\n", i, err)
			continue
		}

		// verify result size
		if uint32(len(ids)) != k || uint32(len(distances)) != k {
			fmt.Printf("Unexpected result size for query %d\n", i)
			continue
		}

		// calculate recall
		correct := 0
		groundTruthSet := make(map[uint32]struct{})
		for j := 0; j < int(k); j++ {
			groundTruthSet[exactDistances[j].id] = struct{}{}
		}
		for _, id := range ids {
			if _, exists := groundTruthSet[id]; exists {
				correct++
			}
		}

		recall := float64(correct) / float64(k)
		totalRecall += recall
		fmt.Printf("Query %d recall: %.2f\n", i, recall)
	}

	// calculate and verify average recall
	averageRecall := totalRecall / float64(numQueries)
	if averageRecall < 0.8 {
		fmt.Println("Average recall too low:", averageRecall)
		return
	}
	fmt.Printf("Average recall success: %.2f\n", averageRecall)

	// save index
	index.SaveIndex("../index.bin")
}

func TestLoadIndex() {
	index, err := vamana.LoadIndex("../index.bin")
	if err != nil {
		fmt.Println("Failed to load index:", err)
		return
	}
	defer index.Free()
}

func main() {
	TestVamanaBasic()
	TestLoadIndex()
}
