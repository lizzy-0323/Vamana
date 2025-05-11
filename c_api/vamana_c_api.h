#ifndef VAMANA_C_API_H
#define VAMANA_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque handle for the Vamana index
typedef struct VamanaIndex VamanaIndex;

// Create a new Vamana index
VamanaIndex *vamana_create_index(uint32_t dimension, uint32_t max_points,
                                 float alpha, uint32_t R, uint32_t L);

// Add a point to the index
int vamana_add_point(VamanaIndex *index, const float *point, uint32_t id);

// Build the index (optimize connections)
int vamana_build_index(VamanaIndex *index);

// Search for nearest neighbors
int vamana_search(VamanaIndex *index, const float *query, uint32_t k,
                  uint32_t ef_search, uint32_t *ids, float *distances);

// Delete the index and free memory
void vamana_free_index(VamanaIndex *index);

// Save index to file
int vamana_save_index(VamanaIndex *index, const char *path);

// Load index from file
VamanaIndex *vamana_load_index(const char *path, uint32_t dim);

// Get point data from index
int vamana_get_point(VamanaIndex *index, uint32_t id, float *point);

#ifdef __cplusplus
}
#endif

#endif // VAMANA_C_API_H
