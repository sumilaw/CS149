#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <vector>
#include <atomic>
// #include <omp.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set* list) {
    list->count = 0;
}

void vertex_set_init(vertex_set* list, int count) {
    list->max_vertices = count;
    list->vertices = (int*)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
int top_down_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int *distances)
{
    int new_frontier_edge = 0;
    #pragma omp parallel 
    {
        std::vector<int> vec;
        int edge_count = 0;
        #pragma omp for
        for (int i=0; i<frontier->count; i++) {

            int node = frontier->vertices[i];

            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                            ? g->num_edges
                            : g->outgoing_starts[node + 1];
            
            int desired = distances[node] + 1;
            // attempt to add all neighbors to the new frontier
            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int outgoing = g->outgoing_edges[neighbor];

                if (distances[outgoing] == NOT_VISITED_MARKER && 
                    __sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, desired)) {
                    vec.emplace_back(outgoing);
                    edge_count += outgoing_size(g, outgoing);
                }
            }
        }
        int start_pos = __sync_fetch_and_add(&new_frontier->count, vec.size());
        __sync_fetch_and_add(&new_frontier_edge, edge_count);
        std::copy(vec.begin(), vec.end(), new_frontier->vertices + start_pos);
    }
    return new_frontier_edge;
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution* sol) {

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0) {
// #define VERBOSE
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
#undef VERBOSE

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}

int bottom_up_step(
    Graph g,
    vertex_set* frontier,
    vertex_set* new_frontier,
    int *distances,
    int dist)
{
    int new_frontier_edge = 0;
    #pragma omp parallel
    {
        std::vector<int>arr;
        int edge_count = 0;
        #pragma omp for schedule(dynamic, 200)
        for (int i=0; i < g->num_nodes; i++) {

            if (distances[i] != NOT_VISITED_MARKER) {
                continue;
            }

            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1)
                        ? g->num_edges
                        : g->incoming_starts[i + 1];

            for (int neighbor=start_edge; neighbor<end_edge; neighbor++) {
                int incoming = g->incoming_edges[neighbor];
                if (distances[incoming] == dist) {
                    distances[i] = dist + 1;
                    arr.emplace_back(i);
                    edge_count += outgoing_size(g, i);
                    break;
                }
            }
        }

        int start_pos = __sync_fetch_and_add(&new_frontier->count, arr.size());
        __sync_fetch_and_add(&new_frontier_edge, edge_count);
        std::copy(arr.begin(), arr.end(), new_frontier->vertices + start_pos);
    }

    return new_frontier_edge;
}

void bfs_bottom_up(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.

    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int dist = 0;

    while (frontier->count != 0) {
// #define VERBOSE
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        
        vertex_set_clear(new_frontier);
        bottom_up_step(graph, frontier, new_frontier, sol->distances, dist++);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
#undef VERBOSE

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

}

void bfs_hybrid(Graph graph, solution* sol)
{
    // CS149 students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set* frontier = &list1;
    vertex_set* new_frontier = &list2;

    for (int i=0; i<graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER;
    }

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    int dist = 0;
    int next_edge = outgoing_size(graph, ROOT_NODE_ID); // 影响 top to down
    int avg = num_edges(graph) / num_nodes(graph);

    while (frontier->count != 0) {
// #define VERBOSE
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif
        vertex_set_clear(new_frontier);
        if (next_edge > num_nodes(graph)) {
            next_edge = bottom_up_step(graph, frontier, new_frontier, sol->distances, dist);
        } else {
            next_edge = top_down_step(graph, frontier, new_frontier, sol->distances);
        }
        dist += 1;

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif
#undef VERBOSE

        // swap pointers
        vertex_set* tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

}
