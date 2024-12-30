#include "Graph.h"

// Add an edge with a label between nodes u and v
void Graph::addEdge(int u, int v, const std::string& label) {
    adjList[u][v] = label;
    adjList[v][u] = label; // For undirected graph
}

// Set the name for a vertex at a specific position
void Graph::setVertexName(int position, int name) {
    vertexNames[position] = name;
}

// Depth-First Search to find connected components
void Graph::dfs(int node, std::unordered_set<int>& visited, std::unordered_set<int>& component) const {
    visited.insert(node);
    component.insert(node);

    for (const auto& neighbor : adjList.at(node)) {
        if (visited.find(neighbor.first) == visited.end()) {
            dfs(neighbor.first, visited, component);
        }
    }
}

// Helper function to get the connected components of the graph
std::vector<std::unordered_set<int>> Graph::getConnectedComponents() const {
    std::unordered_set<int> visited;
    std::vector<std::unordered_set<int>> components;

    for (const auto& node : adjList) {
        if (visited.find(node.first) == visited.end()) {
            std::unordered_set<int> component;
            dfs(node.first, visited, component);
            components.push_back(component);
        }
    }

    return components;
}

// Generate a subgraph for a given subset of nodes
void Graph::generateSubgraphFromSubset(const std::unordered_set<int>& subset, Graph& subgraph) const {
    for (auto node : subset) {
        for (const auto& neighbor : adjList.at(node)) {
            if (subset.find(neighbor.first) != subset.end()) {
                subgraph.addEdge(node, neighbor.first, neighbor.second);
            }
        }
        if (vertexNames.find(node) != vertexNames.end()) {
            subgraph.setVertexName(node, vertexNames.at(node));
        }
    }
}

// Encode the graph into a string representation
std::string Graph::encode() const {
    std::string encoding;

    std::vector<int> nodes;
    for (const auto& node : adjList) {
        nodes.push_back(node.first);
    }

    std::sort(nodes.begin(), nodes.end(), [this](int a, int b) {
        return vertexNames.at(a) < vertexNames.at(b);
        });

    for (const int& node : nodes) {
        int vertexName = vertexNames.at(node);
        encoding += std::to_string(vertexName) + "(";

        std::vector<std::pair<int, std::string>> neighbors;
        for (const auto& neighbor : adjList.at(node)) {
            neighbors.push_back(neighbor);
        }

        std::sort(neighbors.begin(), neighbors.end(), [this](const std::pair<int, std::string>& a, const std::pair<int, std::string>& b) {
            return vertexNames.at(a.first) < vertexNames.at(b.first);
            });

        for (const auto& neighbor : neighbors) {
            int neighborName = vertexNames.at(neighbor.first);
            encoding += std::to_string(neighborName) + ":" + neighbor.second + ",";
        }

        encoding += ")";
    }

    return encoding;
}

// Calculate the size of the graph in edges
int Graph::size() const {
    int size = 0;
    for (const auto& node : adjList) {
        size += node.second.size();
    }
    return size / 2; // For undirected graph
}

// Calculate the tree center
std::vector<int> Graph::calculateTreeCenter() const {
    std::unordered_map<int, int> degree;
    std::queue<int> leafNodes;

    for (const auto& node : adjList) {
        degree[node.first] = node.second.size();
        if (degree[node.first] == 1) {
            leafNodes.push(node.first);
        }
    }

    int remainingNodes = adjList.size();
    while (remainingNodes > 2) {
        int leafCount = leafNodes.size();
        remainingNodes -= leafCount;

        for (int i = 0; i < leafCount; ++i) {
            int leaf = leafNodes.front();
            leafNodes.pop();

            for (const auto& neighbor : adjList.at(leaf)) {
                degree[neighbor.first]--;
                if (degree[neighbor.first] == 1) {
                    leafNodes.push(neighbor.first);
                }
            }
        }
    }

    std::vector<int> center;
    while (!leafNodes.empty()) {
        center.push_back(leafNodes.front());
        leafNodes.pop();
    }

    return center;
}

// Comparison operators
bool Graph::operator<(const Graph& other) const {
    return this->encode() < other.encode();
}

bool Graph::operator==(const Graph& other) const {
    return this->encode() == other.encode();
}
