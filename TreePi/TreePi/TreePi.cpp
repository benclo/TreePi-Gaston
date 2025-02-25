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
#include <memory>
#include <utility>
#include "Graph.h"
#include "BTreePlus.cpp"
#include <random>
#include <ctime>
#include <chrono>

using namespace std;
using namespace chrono;

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
    size_t operator()(const Graph& g) const {
        return hash<string>{}(g.encode()); // Hash the encoded string
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

        // **Ensure single-edge trees are always retained**
        if (tree.size() == 1) {
            finalTrees.push_back(tree);
            continue; // Skip shrinking logic for single-edge trees
        }

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
            remove_if(
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

// Function to calculate the average size of query graphs (sq)
int calculateAverageQueryGraphSize(const vector<Graph>& queryGraphs) {
    int totalSize = 0;
    for (const auto& graph : queryGraphs) {
        totalSize += graph.size();
    }
    return totalSize / queryGraphs.size();
}

// Function to calculate the average size of graphs in the database (sD)
int calculateAverageDatabaseGraphSize(const vector<Graph>& database) {
    int totalSize = 0;
    for (const auto& graph : database) {
        totalSize += graph.size();
    }
    return totalSize / database.size();
}

// Function to calculate alpha, beta, eta
void calculateAlphaBetaEta(int sq, const vector<Graph>& database, int& alpha, int& eta) {
    // Alpha: set between sq/4 and sq/2 (let's choose sq/4 for now)
    alpha = sq / 4;

    // Eta: set to the minimum of sq and average size of graphs in the database (sD)
    int sD = calculateAverageDatabaseGraphSize(database);
    eta = min(sq, sD);

    // Beta: set based on the frequency of substructures with fewer than 10 edges in the database
    int substructureCount = 0;
    int totalGraphs = 0;
}

class Index {
private:
    Graph tree;

    struct NodeTuple {
        string edgeLabel; // Le
        int nodeLabel;         // Lv
        int nodeId;            // Original node ID

        // Comparison operator for sorting
        bool operator<(const NodeTuple& other) const {
            if (edgeLabel != other.edgeLabel) return edgeLabel < other.edgeLabel;
            if (nodeLabel != other.nodeLabel) return nodeLabel < other.nodeLabel;
            return nodeId < other.nodeId; // Tie-breaking by node ID
        }
    };

    // Mapping of Graph nodes to their corresponding NodeTuple
    map<int, NodeTuple> nodeTuples;

    // Method to initialize the mapping of nodes to NodeTuples
    void initializeNodeTuples() {
        // Calculate tree center(s)
        auto centers = tree.calculateTreeCenter();
        if (centers.empty()) {
            throw runtime_error("Tree is empty or disconnected.");
        }

        // Assume the first center as the root
        int root = centers[0];

        // Initialize tuples with NULL edge labels for all nodes
        for (const auto& [node, _] : tree.adjList) {
            nodeTuples[node] = { "", tree.vertexNames.at(node), node };
        }

        // Set edge labels for non-root nodes
        for (const auto& [node, neighbors] : tree.adjList) {
            for (const auto& [neighbor, edgeLabel] : neighbors) {
                if (node != root && nodeTuples[node].edgeLabel.empty()) {
                    nodeTuples[node].edgeLabel = edgeLabel;
                }
            }
        }

        // Set the root's edge label to NULL
        nodeTuples[root].edgeLabel = "NULL";
    }

public:
    Index() : tree() {}

    // Constructor to initialize the Index with a given Graph (tree)
    explicit Index(const Graph& inputTree) : tree(inputTree) {
        initializeNodeTuples();
    }

    // Method to display the NodeTuples for debugging
    void displayNodeTuples() const {
        cout << "Node Tuples:\n";
        for (const auto& [nodeId, tuple] : nodeTuples) {
            cout << "Node " << nodeId << ", Label (" << tuple.edgeLabel << ", " << tuple.nodeLabel << ")\n";
        }
    }

    // Method to get and display the canonical form
    void displayCanonicalForm() const {
        string canonicalForm = constructCanonicalForm();
        cout << "Canonical Form: " << canonicalForm << endl;
    }

    // Method to display basic information about the tree (for debugging)
    void displayTreeInfo() const {
        cout << "Tree Encoding: " << tree.encode() << endl;
        cout << "Tree Size (edges): " << tree.size() << endl;

        auto centers = tree.calculateTreeCenter();
        cout << "Tree Centers: ";
        for (int center : centers) {
            cout << center << " (Name: " << tree.vertexNames.at(center) << ") ";
        }
        cout << endl;
    }

    // Construct the canonical form of the tree using BFS
    string constructCanonicalForm() const {
        auto centers = tree.calculateTreeCenter();
        if (centers.empty()) {
            return ""; // Handle empty or disconnected trees
        }
        int root = centers[0];

        // BFS traversal
        queue<pair<int, int>> bfsQueue; // Pair of (node, parent)
        bfsQueue.push({ root, -1 });

        stringstream canonicalForm;

        while (!bfsQueue.empty()) {
            int current = bfsQueue.front().first;
            int parent = bfsQueue.front().second;
            bfsQueue.pop();

            // Add the current node's tuple (Le, Lv, parent Lv) to the canonical form
            const NodeTuple& tuple = nodeTuples.at(current);
            canonicalForm << "(" << tuple.edgeLabel << "," << tuple.nodeLabel << ")";

            // Gather and sort children
            vector<NodeTuple> children;
            for (const auto& neighbor : tree.adjList.at(current)) {
                if (neighbor.first != parent) { // Skip the parent node
                    children.push_back({ neighbor.second, tree.vertexNames.at(neighbor.first), neighbor.first });
                }
            }
            sort(children.begin(), children.end());

            // Add children to the BFS queue in sorted order
            for (const auto& child : children) {
                bfsQueue.push({ child.nodeId, current });
            }
        }

        return canonicalForm.str();
    }

    // Add to Index class
    bool operator<(const Index& other) const {
        return tree.encode() < other.tree.encode();
    }

    // Optionally, add other comparison operators for flexibility
    bool operator>(const Index& other) const {
        return tree.encode() > other.tree.encode();
    }

    bool operator==(const Index& other) const {
        return tree.encode() == other.tree.encode();
    }

    bool operator!=(const Index& other) const {
        return !(*this == other);
    }

    friend ostream& operator<<(ostream& os, const Index& index) {
        os << index.constructCanonicalForm() << "\n";
        return os; // Return the stream to allow chaining
    }

};

// Check if a graph is a feature tree by looking it up in the feature tree index
bool isFeatureTree(Index idx, BPlusTree<Index> BTree) {
    if (BTree.search(idx))
        return true;
    else
        return false;
}

// Function to generate chemical graphs and write to a file
void generateChemicalGraphs(int numGraphs = 10, int numVertices = 20, int numEdges = 30) {
    // Initialize random number generator
    mt19937 rng(static_cast<unsigned>(time(0))); // Seed with current time
    uniform_int_distribution<int> vertexDist(1, numVertices);
    uniform_int_distribution<int> labelDist(1, 3); // For vertex labels '1', '2', '3'
    uniform_int_distribution<int> edgeLabelDist(0, 2); // For edge labels 'a', 'b', 'c'

    // Open the output file
    ofstream outFile("graph.txt");

    // Generate the graphs
    for (int g = 0; g < numGraphs; ++g) {
        // Vertex labels: Randomly assign 1, 2, or 3 to each vertex (vertex names are from 1 to numVertices)
        vector<int> vertices(numVertices);
        for (int i = 0; i < numVertices; ++i) {
            vertices[i] = labelDist(rng); // Labels are randomly chosen from {1, 2, 3}
        }

        // Generate edges: Directed edges with cycles, but no loops
        vector<tuple<int, int, string>> edges;
        for (int i = 0; i < numEdges; ++i) {
            int v1 = vertexDist(rng);
            int v2 = vertexDist(rng);

            // Prevent loops (no edge from a vertex to itself)
            while (v1 == v2) {
                v2 = vertexDist(rng);
            }

            string label = (edgeLabelDist(rng) == 0) ? "a" : (edgeLabelDist(rng) == 1) ? "b" : "c";
            edges.push_back(make_tuple(v1, v2, label));
        }

        // Write graph details to the file
        outFile << "g" << g + 1 << ":\n";
        for (const auto& edge : edges) {
            outFile << "edge " << get<0>(edge) << " " << get<1>(edge) << " " << get<2>(edge) << "\n";
        }
        for (int i = 0; i < numVertices; ++i) {
            outFile << "vertex " << i + 1 << " " << vertices[i] << "\n";  // Vertex names are from 1 to numVertices
        }
        outFile << "end\n";
    }

    // Close the output file
    outFile.close();
    cout << "Number of graphs: " << numGraphs << " Number of vertices: " << numVertices << " Number of edges: " << numEdges << endl;
}

int main() {

    chrono::time_point<chrono::system_clock> start, end;

    start = chrono::system_clock::now();
    generateChemicalGraphs();
    end = chrono::system_clock::now();

    duration<double> elapsed_seconds = end - start;

    cout << "Graph generation time:" << elapsed_seconds.count() << endl;

    vector<Graph> database = setupGraphs("graph.txt"); // Setup the graphs

    // Calculate the average size of query graphs (sq)
    int sq = 10; // Example value for sq (you can adjust this)

    // Calculate alpha, beta, eta
    int alpha, beta = 5, eta;
    calculateAlphaBetaEta(sq, database, alpha, eta);

    unordered_map<Graph, unordered_set<int>, GraphHasher> subtreeFrequency = calculateSubtreeFrequency(database); // Calculate subtree frequencies

    vector<Graph> freqTrees = filterTreesBySupport(subtreeFrequency, alpha, beta, eta); // Filter trees based on support function

    double gamma = 2;
    vector<Graph> finalTrees = shrinkTrees(freqTrees, subtreeFrequency, gamma); // Shrink the trees based on intersection

    start = chrono::system_clock::now();

    BPlusTree<Index> BTree(3);

    for (auto tree : finalTrees)
    {
        Index idx(tree);
        BTree.insert(idx);
    }

    end = chrono::system_clock::now();

    elapsed_seconds = end - start;

    cout << "Indexing time:" << elapsed_seconds.count() << endl;

    start = chrono::system_clock::now();

    vector<Graph> queries = setupGraphs("query.txt"); // Setup the graphs

    for (auto query : queries) {
        Index queryIdx(query);
        cout << isFeatureTree(queryIdx, BTree);
    }

    end = chrono::system_clock::now();

    elapsed_seconds = end - start;

    cout << "Querying time:" << elapsed_seconds.count() << endl;
        
    return 0;
}