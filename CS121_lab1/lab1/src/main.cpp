#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include "prefix.hpp"
#include <string>
#include <sstream>
#include <random>

#define LARGE 1000

// thread unsafe mask. Used for quickly judge visited nodes.
inline bool check_mask(unsigned int &mask, int node) {
    if ((mask & (1U << (node % 32))) != 0) {
        return true;
    } else {
        mask |= (1U << (node % 32));
        return false;
    }
}

bool atomic_tas(bool *val) {
    bool expected = false;
    return __atomic_compare_exchange_n(val, &expected, true, false, 
                                      __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
}

// Encapsulated BFS that reuses buffers allocated by the caller.
// Returns elapsed seconds for the run and sets 'visited_count'.
double run_bfs(int start,
               int n,
               const std::vector<std::vector<int>> &nbr,
               const std::vector<int> &dgr,
               std::vector<int> &que,
               std::vector<int> &tmp,
               std::vector<int> &filter_large,
               std::vector<int> &large_node_buf,
               std::vector<int> &small_node_buf,
               std::vector<std::vector<int>> &new_que_buf,
               std::vector<int> &small_lens_buf,
               bool *vis,
               std::vector<unsigned int> &mask,
               std::vector<int> &dist,
               std::vector<int> &parent,
               double &out_large_seconds,
               double &out_small_seconds) {

    // reset per-run arrays
    for (int i = 0; i < n; ++i) {
        vis[i] = false;
        dist[i] = -1;
        parent[i] = -1;
    }
    std::fill(mask.begin(), mask.end(), 0u);

    int len = 1;
    que[0] = start;
    vis[start] = true;
    dist[start] = 0;
    parent[start] = -1;
    mask[start / 32] = (1U << (start % 32));

    out_large_seconds = out_small_seconds = 0.0;

    double t0 = omp_get_wtime();

    while (true) {
        #pragma omp parallel for
        for (int i = 0; i < len; i++) {
            if (dgr[que[i]] >= LARGE ) filter_large[i] = 1;
            else filter_large[i] = 0;
        }
        auto presum_large = prefix_sum(filter_large.data(), len);
        int large_num = presum_large[len - 1];

        // reuse buffers
        std::vector<int> &large_node = large_node_buf;
        std::vector<int> &small_node = small_node_buf;
        #pragma omp parallel for
        for (int i = 0; i < len; i++) {
            if (filter_large[i]) large_node[presum_large[i] - 1] = que[i];
            else small_node[i - presum_large[i]] = que[i];
        }
        

        int offset_large = 0;

        double t_large_start = omp_get_wtime();
        for (int i = 0; i < large_num; i++) {
            int u = large_node[i];
            int num_threads = omp_get_num_threads();
            std::vector<int> thread_lens((size_t)num_threads);

            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int chunk_size = (dgr[u] + num_threads - 1) / num_threads;
                int start_idx = tid * chunk_size;
                int end = (start_idx + chunk_size < dgr[u]) ? start_idx + chunk_size : dgr[u];

                std::vector<int> local_que;

                for (int j = start_idx; j < end; j++) {
                    int v = nbr[u][j];
                    if(!check_mask(mask[v / 32], v)) {
                        if (atomic_tas(&vis[v])) {
                            parent[v] = u;
                            dist[v] = dist[u] + 1;
                            local_que.push_back(v);
                        }
                    }
                }
                thread_lens[tid] = (int)local_que.size();

                if (tid == 0) {
                    #pragma omp barrier
                    int offset = 0;
                    for (int t = 0; t < num_threads; t++) {
                        int temp = thread_lens[t];
                        thread_lens[t] = offset;
                        offset += temp;
                    }
                    offset_large = offset;
                } else {
                    #pragma omp barrier
                }

                for (int j = 0; j < (int)local_que.size(); j++) {
                    tmp[thread_lens[tid] + j] = local_que[j];
                }
            }
        }
        out_large_seconds += omp_get_wtime() - t_large_start;

        int small_num = len - large_num;
        // reuse new_que_buf and small_lens_buf; only use first small_num entries
        #pragma omp parallel for
        for (int i = 0; i < small_num; ++i) {
            new_que_buf[i].clear();
            small_lens_buf[i] = 0;
        }
        

        double t_small_start = omp_get_wtime();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < small_num; i++) {
            int u = small_node[i];
            for (int j = 0; j < dgr[u]; j++) {
                int v = nbr[u][j];
                if(!check_mask(mask[v / 32], v)) {
                    if (atomic_tas(&vis[v])) {
                        parent[v] = u;
                        dist[v] = dist[u] + 1;
                        new_que_buf[i].push_back(v);
                    }
                }
            }
            small_lens_buf[i] = (int)new_que_buf[i].size();
        }
        auto presum_small = prefix_sum(small_lens_buf.data(), small_num);
        #pragma omp parallel for schedule(dynamic, 64)
        for (int i = 0; i < small_num; i++) {
            int offset = presum_small[i] - small_lens_buf[i];
            for (int j = 0; j < small_lens_buf[i]; j++) {
                tmp[offset_large + offset + j] = new_que_buf[i][j];
            }
        }
        int offset_small = presum_small[small_num - 1];
        len = offset_large + offset_small;
        out_small_seconds += omp_get_wtime() - t_small_start;

        if (len == 0) break;
        std::swap(que, tmp);
    }

    double t_end = omp_get_wtime() - t0;

    // compute visited count
    int cnt = 0;
    for (int i = 0; i < n; ++i) if (vis[i]) ++cnt;

    // return elapsed seconds
    return t_end;
}

int main(void) {
    int n, l;

    // Read possible MatrixMarket header. If header/comment lines (start with '%')
    // are present, detect "symmetric" and then read the dims line. Otherwise
    // expect the first non-comment line to contain n m l.
    bool symmetric = false;
    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    if (!line.empty() && line[0] == '%') {
        // header present: check for symmetry in the header line
        std::string lower = line;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower.find("sym") != std::string::npos) symmetric = true;

        // skip remaining comment lines starting with '%'
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line[0] == '%') {
                std::string l2 = line;
                std::transform(l2.begin(), l2.end(), l2.begin(), ::tolower);
                if (l2.find("sym") != std::string::npos) symmetric = true;
                continue;
            }
            break; // 'line' now holds the dims line
        }
        std::istringstream iss(line);
        int m;
        if (!(iss >> n >> m >> l)) return 1;
    } else {
        // first line is not a comment: it should contain dims
        std::istringstream iss(line);
        int m;
        if (!(iss >> n >> m >> l)) {
            // fallback to scanf if parsing failed
            if (sscanf(line.c_str(), "%d %d %d", &n, &m, &l) != 3) {
                if (scanf("%d %d %d", &n, &m, &l) != 3) return 1;
            }
        }
    }
    std::vector<int> row((size_t)l);
    std::vector<int> col((size_t)l);
    std::vector<double> dis((size_t)l);

    int ma = 0;
    for (int i = 0; i < l; ++i){
        if (scanf("%d %d %lf", &row[i], &col[i], &dis[i]) != 3) return 1;
        ma = std::max(ma, std::max(row[i], col[i]));
    }
    // std::cout<<"DEBUG: max node id "<<ma<<std::endl;

    n = ma + 1;
    
    std::vector<int> dgr((size_t)n, 0);
    std::vector<std::vector<int>> nbr((size_t)n);

    std::vector<int> dist((size_t)n, -1);
    std::vector<int> parent((size_t)n, -1);

    // build degree and neighbor lists; if symmetric, build undirected
    for (int i = 0; i < l; ++i){
        dgr[row[i]]++;
        if (symmetric) dgr[col[i]]++;
    }

    for (int i = 0; i < n; ++i){
        nbr[i].reserve((size_t)dgr[i]);
    }

    for (int i = 0; i < l; ++i){
        nbr[row[i]].push_back(col[i]);
        if (symmetric) nbr[col[i]].push_back(row[i]);
    }

    bool *vis = new bool[n];

    for (int i = 0; i < n; ++i){
        vis[i] = false;
    }

    std::vector<unsigned int> mask(((size_t)n + 31) / 32 + 1, 0U);

    std::vector<int> que((size_t)n);
    std::vector<int> tmp((size_t)n);

    std::vector<int> filter_large((size_t)n);

    // Reusable temporaries to avoid repeated allocations inside the loop
    std::vector<int> large_node_buf((size_t)n);
    std::vector<int> small_node_buf((size_t)n);
    std::vector<std::vector<int>> new_que_buf((size_t)n);
    std::vector<int> small_lens_buf((size_t)n);

    // Pick up to 20 random start nodes. n may be huge and most nodes may have degree 0,
    // so sample random indices and retry until a non-zero-degree node is found.
    int nonzero_count = 0;
    for (int i = 0; i < n; ++i) if (dgr[i] > 0) ++nonzero_count;
    if (nonzero_count == 0) {
        std::cerr << "No non-zero-degree nodes found; nothing to run.\n";
        delete[] vis;
        return 0;
    }

    int runs = std::min(20, nonzero_count);
    std::mt19937 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> uid(0, n - 1);

    std::cerr << "Running " << runs << " BFS runs from random starts...\n";

    for (int r = 0; r < runs; ++r) {
        int start_node;
        // draw until we find a node with degree > 0
        do {
            start_node = uid(rng);
        } while (dgr[start_node] == 0);
        double larg, small;
        double elapsed = run_bfs(start_node, n, nbr, dgr,
                    que, tmp, filter_large,
                    large_node_buf, small_node_buf,
                    new_que_buf, small_lens_buf,
                    vis, mask, dist, parent,
                    larg, small);
        // Only print the 'large' and 'small' segment timings; send the final
        // run's output to stdout and earlier runs to stderr.
        int cnt = 0, sum = 0;
        for (int i = 0; i < n; ++i) 
            if (vis[i]) {
                ++cnt;
                sum += dgr[i];
            }
        if (r == runs - 1) {
            printf("%d\n", cnt);
            for (int i = 0; i < n; ++i) {
                if (vis[i]) {
                    printf("%d %d %d\n", i, dist[i], parent[i]);
                }
            }
        }
        std::cerr << "Run " << (r+1) << ": start=" << start_node
                      << " elapsed=" << elapsed << " s"
                      << " large=" << larg << " small=" << small
                      << " count=" << cnt << "edge=" << sum 
                      << " MTFLOPS=" << sum/elapsed/1e6 << "\n";
    }

    delete[] vis;

    return 0;
}