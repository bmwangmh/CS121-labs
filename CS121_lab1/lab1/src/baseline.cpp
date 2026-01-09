#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <string>
#include <sstream>
#include <chrono>
#include <random>
#include <iostream>

int main(void) {
    int n, l;

    // Read possible MatrixMarket header. If header/comment lines (start with '%')
    // are present, skip comments and read the dims line. Otherwise expect the
    // first non-comment line to contain n m l.
    std::string line;
    if (!std::getline(std::cin, line)) return 1;
    if (!line.empty() && line[0] == '%') {
        // skip remaining comment lines starting with '%'
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line[0] == '%') continue;
            break; // 'line' now holds the dims line
        }
        std::istringstream iss(line);
        int m;
        if (!(iss >> n >> m >> l)) return 1;
    } else {
        std::istringstream iss(line);
        int m;
        if (!(iss >> n >> m >> l)) {
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

    n = ma + 1;
    std::vector<int> dgr((size_t)n, 0);
    std::vector<std::vector<int>> nbr((size_t)n);

    std::vector<int> dist((size_t)n, -1);
    std::vector<int> parent((size_t)n, -1);

    for (int i = 0; i < l; ++i){
        dgr[row[i]]++;
    }

    for (int i = 0; i < n; ++i){
        nbr[i].reserve((size_t)dgr[i]);
    }

    for (int i = 0; i < l; ++i){
        nbr[row[i]].push_back(col[i]);
    }

    bool *vis = new bool[n];

    // count non-zero degree nodes
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
        // reset per-run arrays
        for (int i = 0; i < n; ++i) {
            vis[i] = false;
            dist[i] = -1;
            parent[i] = -1;
        }

        int startNode;
        do {
            startNode = uid(rng);
        } while (dgr[startNode] == 0);

        std::queue<int> q;
        vis[startNode] = true;
        dist[startNode] = 0;
        parent[startNode] = -1;
        q.push(startNode);

        auto t0 = std::chrono::high_resolution_clock::now();

        while (!q.empty()) {
            int cur = q.front(); q.pop();
            for (auto y : nbr[cur]) {
                if (!vis[y]) {
                    vis[y] = true;
                    q.push(y);
                    dist[y] = dist[cur] + 1;
                    parent[y] = cur;
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);

        int cnt = 0;
        for (int i = 0; i < n; ++i) if (vis[i]) ++cnt;

        // For earlier runs print summary to stderr; for last run print summary
        // and the visited-node list to stdout (same style as `main.cpp`).
        if (r == runs - 1) {
            // final run: print count and node lines to stdout, and summary also to stdout
            printf("Run %d: start=%d elapsed=%lld ms\n", (r+1), startNode, (long long)dur.count());
            printf("%d\n", cnt);
            for (int i = 0; i < n; ++i) if (vis[i]) printf("%d %d %d\n", i, dist[i], parent[i]);
        } else {
            fprintf(stderr, "Run %d: start=%d elapsed=%lld ms\n", (r+1), startNode, (long long)dur.count());
        }
    }

    delete[] vis;
    return 0;
}
