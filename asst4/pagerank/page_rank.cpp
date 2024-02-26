#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

#include "../common/CycleTimer.h"
#include "../common/graph.h"


// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double* solution, double damping, double convergence)
{


  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  for (int i = 0; i < numNodes; ++i) {
    solution[i] = equal_prob;
  }

  // 处理出所有的没有出度的边
  int num = 0;
  #pragma omp parallel for reduction(+:num)
  for (int i = 0;i < numNodes;i++) {
    if (outgoing_size(g, i) == 0) {
      num += 1;
    }
  }
  std::vector<Vertex> no_outgoing_edges(num);
  for (int i = 0;i < numNodes;i++) {
    if (outgoing_size(g, i) == 0) {
      no_outgoing_edges[--num] = i;
    }
  }

  double a1 = (1.0 - damping) / numNodes;
  double a2 = damping / numNodes;

  auto *score_new = new double[numNodes];
  bool converged = false;
  while (!converged) {
    double all_add = 0;
    #pragma omp parallel for reduction(+:all_add)
    for (const int no_outgoing_edge : no_outgoing_edges) {
      all_add += solution[no_outgoing_edge];
    }
    all_add = all_add * a2 + a1;

    double global_diff = 0;
    #pragma omp parallel for reduction(+:global_diff)
    for (int i = 0;i < numNodes;i++) {
      const Vertex *start = incoming_begin(g, i);
      const Vertex *end = incoming_end(g, i);

      score_new[i] = 0;
      for (const Vertex *edge = start; edge != end; edge++) {
        score_new[i] += solution[*edge] / outgoing_size(g, *edge);
      }
      score_new[i] = damping * score_new[i] + all_add;
      global_diff += abs(score_new[i] - solution[i]);
    }
    converged = (global_diff < convergence);
    num += 1;
    std::swap(score_new, solution);
  }
  if (num & 1) {
    memcpy(score_new, solution, sizeof(double) * numNodes);
    score_new = solution;
  }

  delete[] score_new;

  
  /*
     CS149 students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
