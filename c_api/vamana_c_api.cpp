#include "vamana_c_api.h"
#include "index/vamana.h"
#include <cstdint>
#include <memory>

/*
test:
  g++ -std=c++14 -O3 -g -Xpreprocessor -fopenmp
  -I/opt/homebrew/opt/libomp/include example.cpp vamana_c_api.cpp -o
  vamana_example -L/opt/homebrew/opt/libomp/lib -lomp
*/
struct VamanaIndex {
  std::unique_ptr<vamana::Vamana> alg;
};

VamanaIndex *vamana_create_index(uint32_t dimension, uint32_t max_points,
                                 float alpha, uint32_t R, uint32_t L,
                                 uint32_t efSearch) {
  auto index = new VamanaIndex();
  index->alg = std::make_unique<vamana::Vamana>(dimension, max_points, alpha, R,
                                                L, efSearch);
  return index;
}

int vamana_add_point(VamanaIndex *index, const float *point, uint32_t id) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->AddPoint(point, id);
}

int vamana_build_index(VamanaIndex *index) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->BuildIndex();
}

int vamana_search_with_start_point(VamanaIndex *index, const float *query,
                                   const float *start_point, uint32_t k,
                                   uint32_t *result_ids,
                                   float *result_distances) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->SearchWithStartPoint(query, start_point, k, result_ids,
                                          result_distances);
}
int vamana_search(VamanaIndex *index, const float *query, uint32_t k,
                  uint32_t *result_ids, float *result_distances) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->Search(query, k, result_ids, result_distances);
}

int vamana_get_point(VamanaIndex *index, uint32_t id, float *point) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->GetPoint(id, point);
}

// Save index to file
int vamana_save_index(VamanaIndex *index, const char *path) {
  if (!index || !index->alg) {
    return -1;
  }
  return index->alg->SaveIndex(path);
}

// Load index from file
VamanaIndex *vamana_load_index(const char *path) {
  auto index = new VamanaIndex();
  index->alg = std::make_unique<vamana::Vamana>(path);
  return index;
}

void vamana_print_params(VamanaIndex *index) {
  if (!index || !index->alg) {
    std::cout << "Index not initialized" << std::endl;
    return;
  }

  std::cout << "Vamana Index Parameters:" << std::endl;
  std::cout << "L (search list size): " << index->alg->GetL() << std::endl;
  std::cout << "R (max degree): " << index->alg->GetR() << std::endl;
  std::cout << "Dimension: " << index->alg->GetDimension() << std::endl;
  std::cout << "Alpha: " << index->alg->GetAlpha() << std::endl;
  std::cout << "Max Points: " << index->alg->GetMaxPoints() << std::endl;
  std::cout << "Medoid: " << index->alg->GetMedoid() << std::endl;
  std::cout << "Current Data Size: " << index->alg->GetDataSize() << std::endl;
}

void vamana_free_index(VamanaIndex *index) { delete index; }

uint32_t vamana_get_data_size(VamanaIndex *index) {
  if (!index || !index->alg) {
    return 0;
  }
  return index->alg->GetDataSize();
}