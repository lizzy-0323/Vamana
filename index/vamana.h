#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <queue>
#include <random>
#include <vector>

namespace vamana {

class Vamana {
private:
  uint32_t dimension_;
  uint32_t max_points_;
  float alpha_;
  uint32_t R_; // equals max degree
  uint32_t L_;
  uint32_t efSearch_;
  uint32_t medoid_;
  std::vector<std::vector<float>> points_;
  std::vector<uint32_t> ids_;
  std::vector<std::vector<uint32_t>> graph_;

public:
  Vamana(uint32_t dimension, uint32_t max_points, float alpha, uint32_t R,
         uint32_t L, uint32_t efSearch)
      : dimension_(dimension), max_points_(max_points), alpha_(alpha), R_(R),
        L_(L), efSearch_(efSearch) {
    points_.reserve(max_points);
    ids_.reserve(max_points);
    graph_.reserve(max_points);
    medoid_ = -1;
  }

  uint32_t GetDimension() const { return dimension_; }

  uint32_t GetMaxPoints() const { return max_points_; }

  uint32_t GetMedoid() const { return medoid_; }

  uint32_t GetDataSize() const { return points_.size(); }

  float GetAlpha() const { return alpha_; }

  uint32_t GetR() const { return R_; }

  uint32_t GetL() const { return L_; }

  Vamana(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      throw std::runtime_error("Failed to open file");
    }

    uint32_t n;
    in.read(reinterpret_cast<char *>(&n), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&dimension_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&R_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&L_), sizeof(uint32_t));
    in.read(reinterpret_cast<char *>(&alpha_), sizeof(float));
    in.read(reinterpret_cast<char *>(&medoid_), sizeof(uint32_t));
    max_points_ = n;

    // read data and ids
    points_.reserve(n);
    ids_.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
      std::vector<float> point(dimension_);
      uint32_t id;
      in.read(reinterpret_cast<char *>(point.data()),
              dimension_ * sizeof(float));
      in.read(reinterpret_cast<char *>(&id), sizeof(uint32_t));
      points_.push_back(std::move(point));
      ids_.push_back(id);
    }

    // load graph
    graph_.reserve(max_points_);
    for (uint32_t i = 0; i < max_points_; ++i) {
      uint32_t degree;
      in.read(reinterpret_cast<char *>(&degree), sizeof(uint32_t));
      std::vector<uint32_t> neighbors(degree);
      in.read(reinterpret_cast<char *>(neighbors.data()),
              degree * sizeof(uint32_t));
      graph_.push_back(std::move(neighbors));
    }

    in.close();
  }

  ~Vamana() {}

  // Add a point to the index
  int AddPoint(const float *point, uint32_t id) {
    if (points_.size() >= max_points_) {
      return -1;
    }

    std::vector<float> new_point(point, point + dimension_);
    points_.push_back(std::move(new_point));
    ids_.push_back(id);
    graph_.push_back(std::vector<uint32_t>());
    return 0;
  }
  // Build graph index without parallel
  int BuildIndexWithoutParallel() {
    // * This is temporary made for pybinding
    if (points_.empty()) {
      std::cerr << "No points to build index" << std::endl;
      return -1;
    }

    const uint32_t n = points_.size();
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize R-regular directed graph
    InitializeGraph(n, gen);

    std::cout << "graph init success" << std::endl;

    // Find medoid point
    medoid_ = FindMedoidWithoutParallel();
    // std::cout << "medoid: " << medoid_ << std::endl;
    // std::cout << "find medoid" << std::endl;

    // Generate random node order
    std::vector<uint32_t> perm_list(n);
    std::iota(perm_list.begin(), perm_list.end(), 0);
    std::shuffle(perm_list.begin(), perm_list.end(), gen);
    // std::cout << "generate random node order" << std::endl;

    std::vector<bool> visited(n, false);
    std::priority_queue<std::pair<float, uint32_t>> candidates;
    std::vector<uint32_t> visited_nodes;
    visited_nodes.reserve(n);

    // First round: use alpha=1.0
    for (uint32_t idx = 0; idx < perm_list.size(); ++idx) {
      uint32_t i = perm_list[idx];
      if (i == medoid_)
        continue;

      // Greedy search from medoid
      greedySearch(points_[i].data(), medoid_, visited, visited_nodes,
                   candidates);

      std::vector<uint32_t> neighbors1;
      neighbors1.reserve(R_);
      robustPrune(i, visited_nodes, neighbors1, 1.0f);

      graph_[i] = neighbors1;

      // Handle bidirectional connections
      for (uint32_t j : neighbors1) {
        std::vector<uint32_t> &j_graph = graph_[j];

        if (j_graph.size() >= R_) {
          std::vector<uint32_t> j_visited = visited_nodes;
          j_visited.push_back(i);
          std::vector<uint32_t> j_neighbors;
          j_neighbors.reserve(R_);
          robustPrune(j, j_visited, j_neighbors, 1.0f);
          j_graph = j_neighbors;
        } else {
          auto it = std::find(j_graph.begin(), j_graph.end(), i);
          if (it == j_graph.end()) {
            j_graph.push_back(i);
          }
        }
      }
    }

    // Second round: use alpha=alpha_
    for (uint32_t idx = 0; idx < perm_list.size(); ++idx) {
      uint32_t i = perm_list[idx];
      if (i == medoid_)
        continue;

      // Perform greedy search from medoid
      greedySearch(points_[i].data(), medoid_, visited, visited_nodes,
                   candidates);

      std::vector<uint32_t> neighbors2;
      neighbors2.reserve(R_);
      robustPrune(i, visited_nodes, neighbors2, alpha_);

      graph_[i] = neighbors2;

      // Handle bidirectional connections
      for (uint32_t j : neighbors2) {
        std::vector<uint32_t> &j_graph = graph_[j];

        if (j_graph.size() >= R_) {
          std::vector<uint32_t> j_visited = visited_nodes;
          j_visited.push_back(i);
          std::vector<uint32_t> j_neighbors;
          j_neighbors.reserve(R_);
          robustPrune(j, j_visited, j_neighbors, alpha_);
          j_graph = j_neighbors;
        } else {
          auto it = std::find(j_graph.begin(), j_graph.end(), i);
          if (it == j_graph.end()) {
            j_graph.push_back(i);
          }
        }
      }
    }

    return 0;
  }

  // Build the graph index
  int BuildIndex() {
    if (points_.empty()) {
      std::cerr << "No points to build index" << std::endl;
      return -1;
    }

    const uint32_t n = points_.size();
    std::random_device rd;
    std::mt19937 gen(rd());

    // Initialize R-regular directed graph
    InitializeGraph(n, gen);

    std::cout << "grpah init success" << std::endl;

    // Find medoid point
    medoid_ = FindMedoid();

    // Generate random node order
    std::vector<uint32_t> perm_list(n);
    std::iota(perm_list.begin(), perm_list.end(), 0);
    std::shuffle(perm_list.begin(), perm_list.end(), gen);

#pragma omp parallel
    {
      std::vector<bool> thread_visited(n, false);
      std::priority_queue<std::pair<float, uint32_t>> thread_candidates;
      std::vector<uint32_t> thread_visited_nodes;
      thread_visited_nodes.reserve(n);

      // first round：use alpha=1.0
#pragma omp for schedule(dynamic)
      for (uint32_t idx = 0; idx < perm_list.size(); ++idx) {
        uint32_t i = perm_list[idx];
        if (i == medoid_)
          continue;

        // greedy search from medoid
        greedySearch(points_[i].data(), medoid_, thread_visited,
                     thread_visited_nodes, thread_candidates);

        std::vector<uint32_t> neighbors1;
        neighbors1.reserve(R_);
        robustPrune(i, thread_visited_nodes, neighbors1, 1.0f);

#pragma omp critical
        {
          graph_[i] = neighbors1;

          // Handle bidirectional connections
          for (uint32_t j : neighbors1) {
            std::vector<uint32_t> &j_graph = graph_[j];

            if (j_graph.size() >= R_) {
              std::vector<uint32_t> j_visited = thread_visited_nodes;
              j_visited.push_back(i);
              std::vector<uint32_t> j_neighbors;
              j_neighbors.reserve(R_);
              robustPrune(j, j_visited, j_neighbors, 1.0f);
              j_graph = j_neighbors;
            } else {
              auto it = std::find(j_graph.begin(), j_graph.end(), i);
              if (it == j_graph.end()) {
                j_graph.push_back(i);
              }
            }
          }
        }
      }

      // Second round: use alpha=alpha_
#pragma omp for schedule(dynamic)
      for (uint32_t idx = 0; idx < perm_list.size(); ++idx) {
        uint32_t i = perm_list[idx];
        if (i == medoid_)
          continue;

        // Perform greedy search from medoid
        greedySearch(points_[i].data(), medoid_, thread_visited,
                     thread_visited_nodes, thread_candidates);

        std::vector<uint32_t> neighbors2;
        neighbors2.reserve(R_);
        robustPrune(i, thread_visited_nodes, neighbors2, alpha_);

#pragma omp critical
        {
          graph_[i] = neighbors2;

          // Handle bidirectional connections
          for (uint32_t j : neighbors2) {
            std::vector<uint32_t> &j_graph = graph_[j];

            if (j_graph.size() >= R_) {
              std::vector<uint32_t> j_visited = thread_visited_nodes;
              j_visited.push_back(i);
              std::vector<uint32_t> j_neighbors;
              j_neighbors.reserve(R_);
              robustPrune(j, j_visited, j_neighbors, alpha_);
              j_graph = j_neighbors;
            } else {
              auto it = std::find(j_graph.begin(), j_graph.end(), i);
              if (it == j_graph.end()) {
                j_graph.push_back(i);
              }
            }
          }
        }
      }
    }
    return 0;
  }

  // Initialize R-regular directed graph
  void InitializeGraph(uint32_t n, std::mt19937 &gen) {
    std::vector<std::vector<uint32_t>> temp_graph(n);
    std::vector<uint32_t> out_degree(n, 0);
    std::vector<uint32_t> in_degree(n, 0);
    std::vector<std::vector<bool>> connected(n, std::vector<bool>(n, false));

    for (uint32_t i = 0; i < n; ++i) {
      std::vector<uint32_t> candidates;
      candidates.reserve(n - 1);
      for (uint32_t j = 0; j < n; ++j) {
        if (j != i && !connected[i][j] && in_degree[j] < R_) {
          candidates.push_back(j);
        }
      }

      std::shuffle(candidates.begin(), candidates.end(), gen);
      std::vector<uint32_t> &edges = temp_graph[i];
      edges.reserve(R_);

      for (uint32_t j = 0; j < R_ && j < candidates.size(); ++j) {
        uint32_t target = candidates[j];
        edges.push_back(target);
        connected[i][target] = true;
        ++in_degree[target];
        ++out_degree[i];
      }
    }

    // Ensure the graph is strongly connected
    for (uint32_t i = 0; i < n; ++i) {
      if (in_degree[i] == 0) {
        // Find a node that is not connected to i
        std::vector<uint32_t> possible_sources;
        for (uint32_t j = 0; j < n; ++j) {
          if (j != i && !connected[j][i]) {
            possible_sources.push_back(j);
          }
        }

        if (!possible_sources.empty()) {
          uint32_t random_idx = gen() % possible_sources.size();
          uint32_t random_source = possible_sources[random_idx];
          temp_graph[random_source].push_back(i);
          connected[random_source][i] = true;
          ++in_degree[i];
          ++out_degree[random_source];
        }
      }
    }

    // Move the temporary graph to the final graph
    graph_ = std::move(temp_graph);
  }

  // Save Index
  int SaveIndex(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      std::cerr << "Failed to open file: " << filename << std::endl;
      return -1;
    }

    // write parameters
    uint32_t n = points_.size();
    uint32_t dim = points_[0].size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&R_), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&L_), sizeof(uint32_t));
    out.write(reinterpret_cast<const char *>(&alpha_), sizeof(float));
    out.write(reinterpret_cast<const char *>(&medoid_), sizeof(uint32_t));

    // write data points and ids
    for (uint32_t i = 0; i < n; ++i) {
      out.write(reinterpret_cast<const char *>(points_[i].data()),
                dim * sizeof(float));
      out.write(reinterpret_cast<const char *>(&ids_[i]), sizeof(uint32_t));
    }

    // write graph structure
    for (const auto &neighbors : graph_) {
      uint32_t degree = neighbors.size();
      out.write(reinterpret_cast<const char *>(&degree), sizeof(uint32_t));
      out.write(reinterpret_cast<const char *>(neighbors.data()),
                degree * sizeof(uint32_t));
    }

    out.close();
    return 0;
  }

  int SearchWithStartPoint(const float *query, const float *start_point,
                           uint32_t k, uint32_t *result_ids,
                           float *result_distances) const {
    if (!query || !start_point || !result_ids || !result_distances || k == 0) {
      return -1;
    }

    uint32_t efSearch = efSearch_;
    const uint32_t n = points_.size();
    if (k > n)
      k = n;
    if (efSearch < k)
      efSearch = k; // ef_search equals L here;

    // Find the point in our dataset that matches the start_point
    uint32_t start_idx = 0;
    float min_dist = std::numeric_limits<float>::max();
    for (uint32_t i = 0; i < n; ++i) {
      float dist = ComputeDistance(start_point, points_[i].data());
      if (dist < min_dist) {
        min_dist = dist;
        start_idx = i;
      }
    }

    std::vector<bool> visited(n, false);
    std::priority_queue<std::pair<float, uint32_t>> candidates;
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        results;

    std::vector<uint32_t> visited_nodes;
    visited_nodes.reserve(n);
    greedySearch(query, start_idx, visited, visited_nodes, candidates);

    // Initialize results with visited nodes
    for (uint32_t node : visited_nodes) {
      float dist = ComputeDistance(query, points_[node].data());
      if (results.size() < efSearch || dist < results.top().first) {
        results.emplace(dist, node);
        if (results.size() > efSearch) {
          results.pop();
        }
      }
    }
    // 打印result
    // std::cout << "Result size: " << results.size() << std::endl;

    std::vector<std::pair<float, uint32_t>> final_results;
    final_results.reserve(k);

    while (!results.empty() && final_results.size() < k) {
      final_results.push_back(results.top());
      results.pop();
    }

    for (uint32_t i = 0; i < final_results.size(); ++i) {
      result_distances[i] = final_results[i].first;
      result_ids[i] = ids_[final_results[i].second];
    }

    return final_results.size();
  }

  // Search for k nearest neighbors
  int Search(const float *query, uint32_t k, uint32_t *result_ids,
             float *result_distances) const {
    if (!query || !result_ids || !result_distances || k == 0) {
      return -1;
    }

    uint32_t efSearch = efSearch_;
    const uint32_t n = points_.size();
    if (k > n)
      k = n;
    if (efSearch < k)
      efSearch = k; // ef_search equals L here;

    std::vector<bool> visited(n, false);
    std::priority_queue<std::pair<float, uint32_t>> candidates;
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        results;

    std::vector<uint32_t> visited_nodes;
    visited_nodes.reserve(n);
    greedySearch(query, medoid_, visited, visited_nodes, candidates);

    // 打印visited nodes
    // std::cout << "Visited nodes: " << visited_nodes.size() << std::endl;

    // Initialize results with visited nodes
    for (uint32_t node : visited_nodes) {
      float dist = ComputeDistance(query, points_[node].data());
      if (results.size() < efSearch || dist < results.top().first) {
        results.emplace(dist, node);
        if (results.size() > efSearch) {
          results.pop();
        }
      }
    }
    // 打印result
    // std::cout << "Result size: " << results.size() << std::endl;

    std::vector<std::pair<float, uint32_t>> final_results;
    final_results.reserve(k);

    while (!results.empty() && final_results.size() < k) {
      final_results.push_back(results.top());
      results.pop();
    }

    for (uint32_t i = 0; i < final_results.size(); ++i) {
      result_distances[i] = final_results[i].first;
      result_ids[i] = ids_[final_results[i].second];
    }

    return final_results.size();
  }

  // Get a point by its internal index
  int GetPoint(uint32_t id, float *point) const {
    if (id >= points_.size() || !point) {
      return -1;
    }
    std::copy(points_[id].begin(), points_[id].end(), point);
    return 0;
  }

private:
  // Greedy search from start point
  void greedySearch(
      const float *query_point, uint32_t start_point,
      std::vector<bool> &visited, std::vector<uint32_t> &visited_nodes,
      std::priority_queue<std::pair<float, uint32_t>> &candidates) const {

    // here use the single thread version to find the medoid
    // std::cout << "start_point: " << start_point << std::endl;
    if (start_point == -1) {
      start_point = FindMedoidWithoutParallel();
    }
    const uint32_t n = points_.size();
    visited.assign(n, false);
    while (!candidates.empty())
      candidates.pop();
    visited_nodes.clear();

    // Maintain a search list L, sorted by distance to the query point
    std::vector<std::pair<float, uint32_t>> L;
    L.reserve(L_);

    // Add the start point
    float start_dist =
        ComputeDistance(query_point, points_[start_point].data());
    L.emplace_back(start_dist, start_point);

    while (true) {
      // Find the nearest unvisited point in L
      uint32_t current = n;
      float min_dist = std::numeric_limits<float>::max();
      size_t current_idx = 0;

      for (size_t i = 0; i < L.size(); ++i) {
        if (!visited[L[i].second] && L[i].first < min_dist) {
          min_dist = L[i].first;
          current = L[i].second;
          current_idx = i;
        }
      }

      // If no unvisited point is found, end the search
      if (current == n)
        break;

      // Mark the current point as visited
      visited[current] = true;
      visited_nodes.push_back(current);

      // Add the neighbors of the current point to L
      // std::cout << "Graph neighbors size:" << graph_[current].size()
      //           << std::endl;
      for (uint32_t neighbor : graph_[current]) {
        if (!visited[neighbor]) {
          float dist = ComputeDistance(query_point, points_[neighbor].data());
          L.emplace_back(dist, neighbor);
        }
      }

      // If L exceeds the limit, keep the L_ nearest points
      if (L.size() > L_) {
        std::partial_sort(
            L.begin(), L.begin() + L_, L.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
        L.resize(L_);
      }
    }

    // Add L points to candidates in sorted order
    for (const auto &pair : L) {
      candidates.emplace(-pair.first, pair.second);
    }
  }

  uint32_t FindMedoidWithoutParallel() const {
    // * This is temporary made for pybinding
    const uint32_t n = points_.size();
    if (n == 0)
      return 0;

    uint32_t best_medoid = 0;
    float min_total_dist = std::numeric_limits<float>::max();

    for (uint32_t i = 0; i < n; ++i) {
      float total_dist = 0;
      for (uint32_t j = 0; j < n; ++j) {
        if (i != j) {
          total_dist += ComputeDistance(points_[i].data(), points_[j].data());
        }
      }
      if (total_dist < min_total_dist) {
        min_total_dist = total_dist;
        best_medoid = i;
      }
    }

    return best_medoid;
  }

  uint32_t FindMedoid() const {
    const uint32_t n = points_.size();
    if (n == 0)
      return 0;

    uint32_t best_medoid = 0;
    float min_total_dist = std::numeric_limits<float>::max();

#pragma omp parallel
    {
      uint32_t local_best_medoid = 0;
      float local_min_total_dist = std::numeric_limits<float>::max();

#pragma omp for schedule(dynamic)
      for (uint32_t i = 0; i < n; ++i) {
        float total_dist = 0;
        for (uint32_t j = 0; j < n; ++j) {
          if (i != j) {
            total_dist += ComputeDistance(points_[i].data(), points_[j].data());
          }
        }
        if (total_dist < local_min_total_dist) {
          local_min_total_dist = total_dist;
          local_best_medoid = i;
        }
      }

#pragma omp critical
      {
        if (local_min_total_dist < min_total_dist) {
          min_total_dist = local_min_total_dist;
          best_medoid = local_best_medoid;
        }
      }
    }
    return best_medoid;
  }

  float ComputeDistance(const float *a, const float *b) const {
    // TODO: add inner product
    float dist = 0.0f;
    for (uint32_t i = 0; i < dimension_; ++i) {
      float diff = a[i] - b[i];
      dist += diff * diff;
    }
    return dist;
  }

  // check if the degree of each node is less than max_degree_
  void healthCheck() {
    std::vector<unsigned int> degrees(graph_.size(), 0);
    for (uint32_t i = 0; i < graph_.size(); ++i) {
      degrees[i] += graph_[i].size();
    }

    // use top 10 to check
    for (uint32_t i = 0; i < 10; ++i) {
      if (degrees[i] > R_) {
        std::cerr << "Node " << i << " has degree " << degrees[i]
                  << " which is greater than max degree " << R_ << std::endl;
      } else {
        std::cout << "Node " << i << " Degree " << degrees[i] << std::endl;
      }
    }
  }

  void robustPrune(uint32_t point_id,
                   const std::vector<uint32_t> &visited_nodes,
                   std::vector<uint32_t> &neighbors, float alpha) const {
    if (visited_nodes.empty() || point_id >= points_.size())
      return;

    float min_dist = std::numeric_limits<float>::max();
    int p_star = -1;
    for (uint32_t node : visited_nodes) {
      if (node >= points_.size() || node == point_id)
        continue;
      float dist =
          ComputeDistance(points_[point_id].data(), points_[node].data());
      if (dist < min_dist) {
        min_dist = dist;
        p_star = node;
      }
    }

    // Add the nearest point
    neighbors.push_back(p_star);

    // Use alpha rule to select remaining points
    for (const auto &candidate : visited_nodes) {
      if (candidate >= points_.size() || candidate == point_id)
        continue;
      float dist_p_star =
          ComputeDistance(points_[p_star].data(), points_[candidate].data());
      float dist_p =
          ComputeDistance(points_[point_id].data(), points_[candidate].data());
      if (dist_p_star * alpha >= dist_p &&
          std::find(neighbors.begin(), neighbors.end(), candidate) ==
              neighbors.end()) {
        neighbors.push_back(candidate);
        if (neighbors.size() >= R_)
          break;
      }
    }
  }
};

} // namespace vamana