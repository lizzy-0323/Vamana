#include "c_api/vamana_c_api.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

// generate random vector
std::vector<float> generate_random_vector(uint32_t dimension) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

  std::vector<float> vec(dimension);
  for (uint32_t i = 0; i < dimension; ++i) {
    vec[i] = dis(gen);
  }
  return vec;
}

// compute_distance for calculating ground truth
float compute_distance(const float *a, const float *b, uint32_t dimension) {
  float sum = 0.0f;
  for (uint32_t i = 0; i < dimension; ++i) {
    float diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

int main() {
  // setting parameters
  const uint32_t dimension = 128;    // vector dimension
  const uint32_t num_points = 1000; // number of points
  const uint32_t R = 128; // maximum degree, larger to obtain better connectivity, R in vamana paper
  const uint32_t L = 100;
  const float alpha = 1.2f;       // robust prune parameter, larger to obtain better approximation
  const uint32_t k = 10;          // top-k
  const uint32_t ef_search = 400; // candidate list size

  VamanaIndex *index = vamana_create_index(dimension, num_points, alpha, R, L);
  // generate and add random points
  for (uint32_t i = 0; i < num_points; ++i) {
    auto point = generate_random_vector(dimension);
    if (vamana_add_point(index, point.data(), i) != 0) {
      std::cerr << "Failed to add point " << i << std::endl;
      vamana_free_index(index);
      return -1;
    }
  }
  // build index
  std::cout << "Building index..." << std::endl;
  auto start_time = std::chrono::high_resolution_clock::now();
  if (vamana_build_index(index) != 0) {
    std::cerr << "Failed to build index" << std::endl;
    vamana_free_index(index);
    return -1;
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();
  std::cout << "Index built in " << build_time << " ms" << std::endl;

  // save index
  if (vamana_save_index(index, "../index.bin") != 0) {
    std::cerr << "Failed to save index" << std::endl;
    vamana_free_index(index);
    return -1;
  }

  // Load index
  VamanaIndex *indexLoaded = vamana_load_index("../index.bin");
  if (!indexLoaded) {
    std::cerr << "Failed to load index" << std::endl;
    return -1;
  }

  // generate multiple queries and search
  const int num_queries = 100;
  std::cout << "\nPerforming search test with " << num_queries << " queries..."
            << std::endl;

  std::vector<std::vector<float>> queries;
  queries.reserve(num_queries);
  for (int q = 0; q < num_queries; ++q) {
    queries.push_back(generate_random_vector(dimension));
  }

  std::vector<uint32_t> result_ids(k);
  std::vector<float> result_distances(k);
  std::vector<float> point_buffer(dimension);
  double total_recall = 0.0;
  long total_search_time = 0;

  for (int q = 0; q < num_queries; ++q) {
    const auto &query = queries[q];

    // calculate ground truth
    std::vector<std::pair<float, uint32_t>> exact_distances;
    exact_distances.reserve(num_points);
    for (uint32_t i = 0; i < num_points; ++i) {
      if (vamana_get_point(indexLoaded, i, point_buffer.data()) == 0) {
        float dist =
            compute_distance(query.data(), point_buffer.data(), dimension);
        exact_distances.emplace_back(dist, i);
      }
    }
    std::sort(exact_distances.begin(), exact_distances.end());
    std::vector<uint32_t> ground_truth_ids(k);
    for (uint32_t i = 0; i < k; ++i) {
      ground_truth_ids[i] = exact_distances[i].second;
    }

    // search
    start_time = std::chrono::high_resolution_clock::now();
    int num_results = vamana_search(indexLoaded, query.data(), k, ef_search,
                                    result_ids.data(), result_distances.data());
    end_time = std::chrono::high_resolution_clock::now();
    auto search_time = std::chrono::duration_cast<std::chrono::microseconds>(
                           end_time - start_time)
                           .count();
    total_search_time += search_time;

    if (num_results < 0) {
      std::cerr << "Search failed for query " << q << std::endl;
      continue;
    }

    // calculate recall
    int matches = 0;
    std::unordered_set<uint32_t> result_set(result_ids.begin(),
                                            result_ids.end());
    for (uint32_t i = 0; i < k; ++i) {
      if (result_set.count(ground_truth_ids[i])) {
        matches++;
      }
    }
    float recall = static_cast<float>(matches) / k;
    total_recall += recall;

    // print progress
    if ((q + 1) % 10 == 0) {
      std::cout << "Processed " << (q + 1) << " queries..." << std::endl;
    }
  }

  // print average performance metrics
  double avg_recall = total_recall / num_queries;
  double avg_search_time = static_cast<double>(total_search_time) / num_queries;

  std::cout << "\nPerformance Metrics (averaged over " << num_queries
            << " queries):" << std::endl;
  std::cout << "Average Recall@" << k << ": " << std::fixed
            << std::setprecision(4) << avg_recall * 100 << "%" << std::endl;
  std::cout << "Average Search Time: " << std::fixed << std::setprecision(3)
            << avg_search_time / 1000.0 << " ms" << std::endl;

  // free memory
  vamana_free_index(indexLoaded);
  vamana_free_index(index); 
  return 0;
}
