#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <algorithm>
#include <stack>
#include <queue>
#include <fstream>

using namespace std;

// Graph representation using adjacency list (ordered map)
class Graph {
public:
    map<int, map<int, string>> adjList; // adjacency list with edge labels
    map<int, int> vertexNames; // Map positions to vertex names

    // Add an edge with a label between nodes u and v
    void addEdge(int u, int v, const string& label) {
        adjList[u][v] = label;
        adjList[v][u] = label; // For undirected graph
    }

    // Set the name for a vertex at a specific position
    void setVertexName(int position, int name) {
        vertexNames[position] = name;
    }

    // Helper function to get the connected components of the graph
    vector<unordered_set<int>> getConnectedComponents() const{
        unordered_set<int> visited;
        vector<unordered_set<int>> components;

        for (const auto& node : adjList) {
            if (visited.find(node.first) == visited.end()) {
                unordered_set<int> component;
                dfs(node.first, visited, component);
                components.push_back(component);
            }
        }

        return components;
    }

    // Depth-First Search to find connected components
    void dfs(int node, unordered_set<int>& visited, unordered_set<int>& component) const{
        visited.insert(node);
        component.insert(node);

        for (const auto& neighbor : adjList.at(node)) {
            if (visited.find(neighbor.first) == visited.end()) {
                dfs(neighbor.first, visited, component);
            }
        }
    }

    // Generate a subgraph for a given subset of nodes
    void generateSubgraphFromSubset(const unordered_set<int>& subset, Graph& subgraph) const {
        for (auto node : subset) {
            // Add edges for the current node that connect it to other nodes in the subset
            for (const auto& neighbor : adjList.at(node)) {
                if (subset.find(neighbor.first) != subset.end()) { // Only consider edges within the subset
                    subgraph.addEdge(node, neighbor.first, neighbor.second);
                }
            }
            // Set the vertex name for nodes in the subgraph
            if (vertexNames.find(node) != vertexNames.end()) {
                subgraph.setVertexName(node, vertexNames.at(node));
            }
        }
    }


    string encode() const {
        string encoding = "";

        // Iterate through the nodes in sorted order by vertex name
        vector<int> nodes;
        for (const auto& node : adjList) {
            nodes.push_back(node.first);
        }

        // Sort nodes by their vertex names
        sort(nodes.begin(), nodes.end(), [this](int a, int b) {
            return vertexNames.at(a) < vertexNames.at(b);
            });

        // For each node, sort its neighbors and edge labels
        for (const int& node : nodes) {
            int vertexName = vertexNames.at(node); // Get the vertex name
            encoding += to_string(vertexName) + "(";

            // Sort the neighbors by their node ID to ensure consistency
            vector<pair<int, string>> neighbors;
            for (const auto& neighbor : adjList.at(node)) {
                neighbors.push_back(neighbor);
            }

            // Sort neighbors by their vertex names, not node IDs
            sort(neighbors.begin(), neighbors.end(), [this](const pair<int, string>& a, const pair<int, string>& b) {
                return vertexNames.at(a.first) < vertexNames.at(b.first);
                });

            // Append sorted neighbors and their edge labels to the encoding
            for (const auto& neighbor : neighbors) {
                int neighborName = vertexNames.at(neighbor.first); // Use vertex names
                encoding += to_string(neighborName) + ":" + neighbor.second + ",";
            }

            encoding += ")";  // Close the node's encoding
        }

        return encoding;
    }

    // Method to calculate the size of the graph in edges
    int size() const {
        int size = 0;
        // Iterate over all nodes in the adjacency list
        for (const auto& node : adjList) {
            size += node.second.size();  // Add the number of edges for each node
        }
        return size / 2;  // Since it's an undirected graph, divide by 2
    }

    vector<int> calculateTreeCenter() const {
        // Map to store the degree of each node
        unordered_map<int, int> degree;
        queue<int> leafNodes;

        // Initialize the degree map and identify leaf nodes
        for (const auto& node : adjList) {
            degree[node.first] = node.second.size();
            if (degree[node.first] == 1) {
                leafNodes.push(node.first);
            }
        }

        // Iteratively remove leaf nodes
        int remainingNodes = adjList.size();
        while (remainingNodes > 2) {
            int leafCount = leafNodes.size();
            remainingNodes -= leafCount;

            for (int i = 0; i < leafCount; ++i) {
                int leaf = leafNodes.front();
                leafNodes.pop();

                // Use `at()` for const access to adjList
                for (const auto& neighbor : adjList.at(leaf)) {
                    degree[neighbor.first]--;
                    if (degree[neighbor.first] == 1) {
                        leafNodes.push(neighbor.first);
                    }
                }
            }
        }

        // Remaining nodes are the center
        vector<int> center;
        while (!leafNodes.empty()) {
            center.push_back(leafNodes.front());
            leafNodes.pop();
        }
        return center;
    }

    bool operator<(const Graph& other) const {
        return this->encode() < other.encode(); // Compare based on encoded string
    }

    bool operator==(const Graph& other) const {
        return this->encode() == other.encode(); // Compare based on encoded string
    }
};

// Function to calculate the number of edges in the graph
int calculateSize(const Graph& graph) {
    int size = 0;
    for (const auto& node : graph.adjList) {
        size += node.second.size();
    }
    return size / 2; // Since it's an undirected graph, divide by 2
}

// Helper function to check if a subgraph is connected
bool isConnected(const Graph& subgraph) {
    if (subgraph.adjList.empty()) return false;

    unordered_set<int> visited;
    vector<int> nodes;
    for (const auto& node : subgraph.adjList) {
        nodes.push_back(node.first);
    }

    // Perform DFS starting from the first node
    stack<int> stack;
    stack.push(nodes[0]);
    visited.insert(nodes[0]);

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        for (const auto& neighbor : subgraph.adjList.at(current)) {
            if (visited.find(neighbor.first) == visited.end()) {
                visited.insert(neighbor.first);
                stack.push(neighbor.first);
            }
        }
    }

    // If visited all nodes, the graph is connected
    return visited.size() == nodes.size();
}

// DFS to check for cycles
bool isAcyclicDFS(const Graph& subgraph, int node, int parent, unordered_set<int>& visited) {
    visited.insert(node);

    for (const auto& neighbor : subgraph.adjList.at(node)) {
        if (neighbor.first != parent) {
            if (visited.find(neighbor.first) != visited.end() ||
                !isAcyclicDFS(subgraph, neighbor.first, node, visited)) {
                return false;
            }
        }
    }
    return true;
}

// Helper function to check if a subgraph is acyclic
bool isAcyclic(const Graph& subgraph) {
    unordered_set<int> visited;
    for (const auto& node : subgraph.adjList) {
        if (visited.find(node.first) == visited.end()) {
            if (!isAcyclicDFS(subgraph, node.first, -1, visited)) {
                return false;
            }
        }
    }
    return true;
}

// Generate all valid trees (connected, acyclic subgraphs)
vector<Graph> generateTrees(const unordered_set<int>& component, const Graph& originalGraph) {
    vector<Graph> subtrees;
    vector<int> nodes(component.begin(), component.end());

    for (size_t i = 0; i < (static_cast<unsigned long long>(1) << nodes.size()); ++i) {
        unordered_set<int> subset;
        for (size_t j = 0; j < nodes.size(); ++j) {
            if (i & (static_cast<unsigned long long>(1) << j)) {
                subset.insert(nodes[j]);
            }
        }

        if (!subset.empty()) {
            Graph subgraph;
            originalGraph.generateSubgraphFromSubset(subset, subgraph);

            if (isConnected(subgraph) && isAcyclic(subgraph)) {
                subtrees.push_back(subgraph);
            }
        }
    }

    return subtrees;
}

int supportFunction(int size, int alpha, int beta, int eta) {
    if (size <= alpha) return 1;
    if (size > eta) return INT_MAX; // Exclude large trees
    return 1 + beta * (size - alpha);
}

// Hash function for Graph
struct GraphHasher {
    std::size_t operator()(const Graph& g) const {
        return std::hash<std::string>{}(g.encode()); // Hash the encoded string
    }
};

// Function to read graphs from a file
vector<Graph> setupGraphs(const string& filename) {
    vector<Graph> database;
    ifstream inputFile(filename);
    string line;

    if (!inputFile.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return database;
    }

    Graph currentGraph;
    while (getline(inputFile, line)) {
        // Trim whitespace
        line.erase(remove(line.begin(), line.end(), '\r'), line.end());
        line.erase(remove(line.begin(), line.end(), '\n'), line.end());

        if (line.empty()) continue;

        if (line.back() == ':') { // Start of a new graph
            if (!currentGraph.adjList.empty()) {
                database.push_back(currentGraph);
                currentGraph = Graph();
            }
        }
        else if (line == "end") { // End of the current graph
            if (!currentGraph.adjList.empty()) {
                database.push_back(currentGraph);
                currentGraph = Graph();
            }
        }
        else if (line.find("edge") == 0) { // Parse an edge
            istringstream iss(line);
            string command;
            int u, v;
            string label;
            iss >> command >> u >> v >> label;
            currentGraph.addEdge(u, v, label);
        }
        else if (line.find("vertex") == 0) { // Parse a vertex
            istringstream iss(line);
            string command;
            int position, name;
            iss >> command >> position >> name;
            currentGraph.setVertexName(position, name);
        }
    }

    // Add the last graph if not already added
    if (!currentGraph.adjList.empty()) {
        database.push_back(currentGraph);
    }

    inputFile.close();
    return database;
}

// Function to calculate the frequency of each subtree in the graph database
unordered_map<Graph, unordered_set<int>, GraphHasher> calculateSubtreeFrequency(const vector<Graph>& database) {
    unordered_map<Graph, unordered_set<int>, GraphHasher> subtreeFrequency;

    // Output all subtrees (trees) and track their frequencies across graphs
    for (size_t idx = 0; idx < database.size(); ++idx) {
        auto& graph = database[idx];
        vector<unordered_set<int>> components = graph.getConnectedComponents();

        // Collect all trees for the current graph
        for (auto& component : components) {
            vector<Graph> componentTrees = generateTrees(component, graph);

            // Track each tree's appearance in the graphs
            for (const auto& tree : componentTrees) {
                subtreeFrequency[tree].insert(idx); // Insert the graph index where the tree was found   
            }
        }
    }

    return subtreeFrequency;
}

// Function to filter trees based on the support function
vector<Graph> filterTreesBySupport(const unordered_map<Graph, unordered_set<int>, GraphHasher>& subtreeFrequency, int alpha, int beta, int eta) {
    vector<Graph> freqTrees;

    for (const auto& entry : subtreeFrequency) {
        int support = entry.second.size();
        if (support >= supportFunction(entry.first.size(), alpha, beta, eta)) {
            freqTrees.push_back(entry.first);
        }
    }

    return freqTrees;
}

// Function to shrink trees based on the intersection of support sets
vector<Graph> shrinkTrees(const vector<Graph>& freqTrees, const unordered_map<Graph, unordered_set<int>, GraphHasher>& subtreeFrequency, double gamma) {
    vector<Graph> finalTrees;

    for (size_t idx = 0; idx < freqTrees.size(); ++idx) {
        auto& tree = freqTrees[idx];

        // Extract the connected components of the current tree
        vector<unordered_set<int>> components = tree.getConnectedComponents();

        // Collect all subtrees for the current tree
        vector<Graph> subtrees;
        for (auto& component : components) {
            vector<Graph> componentTrees = generateTrees(component, tree);
            subtrees.insert(subtrees.end(), componentTrees.begin(), componentTrees.end());
        }

        // Remove the tree itself (r) from the list of subtrees
        subtrees.erase(
            std::remove_if(
                subtrees.begin(), subtrees.end(),
                [&tree](const Graph& subtree) { return subtree.encode() == tree.encode(); }
            ),
            subtrees.end()
        );

        // Compute the intersection of support sets of all subtrees
        if (!subtrees.empty()) { // Check to avoid accessing invalid indices
            unordered_set<int> intersection = subtreeFrequency.at(subtrees[0]);
            for (size_t i = 1; i < subtrees.size(); ++i) {
                unordered_set<int> temp;
                for (int idx : intersection) {
                    if (subtreeFrequency.at(subtrees[i]).count(idx)) {
                        temp.insert(idx);
                    }
                }
                intersection = temp; // Update intersection
            }

            // Calculate shrink ratio
            int intersectionSize = intersection.size();
            int treeSupport = subtreeFrequency.at(tree).size();
            double shrinkRatio = static_cast<double>(intersectionSize) / treeSupport;

            // Apply shrinking criterion
            if (shrinkRatio >= gamma) {
                finalTrees.push_back(tree); // Keep the tree
            }
        }
    }

    return finalTrees;
}

// Function to output the final trees and their centers
void outputFinalTrees(const vector<Graph>& finalTrees) {
    cout << "Final Trees after Shrinking:\n";
    for (const auto& tree : finalTrees) {
        cout << tree.encode() << "\n";
    }
}

class Index {
private:
    Graph tree;

    struct NodeTuple {
        string edgeLabel;      // Le
        int nodeLabel;         // Lv
        int nodeId;            // Original node ID

        // Comparison operator for sorting
        bool operator<(const NodeTuple& other) const {
            if (edgeLabel != other.edgeLabel) return edgeLabel < other.edgeLabel;
            if (nodeLabel != other.nodeLabel) return nodeLabel < other.nodeLabel;
            return nodeId < other.nodeId; // Tie-breaking by node ID
        }
    };

public:
    // Constructor to initialize the Index with a given Graph (tree)
    explicit Index(const Graph& inputTree) : tree(inputTree) {}

    // Method to display basic information about the tree (for debugging)
    void displayTreeInfo() const {
        cout << "Tree Encoding: " << tree.encode() << endl;
        cout << "Tree Size (edges): " << tree.size() << endl;

        auto centers = tree.calculateTreeCenter();
        cout << "Tree Centers: ";
        for (int center : centers) {
            cout << center << " (Name: " << tree.vertexNames.at(center) << ") ";
        }
        cout << std::endl;
    }
};

// #TODO
// 1. Index class
// 2. Querying
// 3. Calculation of alpha, beta, eta and gamma

int main() {
    vector<Graph> database = setupGraphs("graph.txt"); // Setup the graphs

    unordered_map<Graph, unordered_set<int>, GraphHasher> subtreeFrequency = calculateSubtreeFrequency(database); // Calculate subtree frequencies

    int alpha = 1, beta = 1, eta = 10;
    vector<Graph> freqTrees = filterTreesBySupport(subtreeFrequency, alpha, beta, eta); // Filter trees based on support function

    double gamma = 1;
    vector<Graph> finalTrees = shrinkTrees(freqTrees, subtreeFrequency, gamma); // Shrink the trees based on intersection

    outputFinalTrees(finalTrees); // Output the final trees

    /*Index idx(finalTrees.front());
    idx.displayTreeInfo();*/

    return 0;
}

