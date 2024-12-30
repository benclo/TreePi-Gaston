#ifndef GRAPH_H
#define GRAPH_H

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <stdexcept>

class Graph {
public:
    std::map<int, std::map<int, std::string>> adjList; // adjacency list with edge labels
    std::map<int, int> vertexNames;                   // map positions to vertex names

    void addEdge(int u, int v, const std::string& label);
    void setVertexName(int position, int name);

    std::vector<std::unordered_set<int>> getConnectedComponents() const;
    void generateSubgraphFromSubset(const std::unordered_set<int>& subset, Graph& subgraph) const;

    std::string encode() const;
    int size() const;
    std::vector<int> calculateTreeCenter() const;

    bool operator<(const Graph& other) const;
    bool operator==(const Graph& other) const;

private:
    void dfs(int node, std::unordered_set<int>& visited, std::unordered_set<int>& component) const;
};

#endif // GRAPH_H
#pragma once
