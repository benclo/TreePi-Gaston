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
    vector<unordered_set<int>> getConnectedComponents() const {
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
    void dfs(int node, unordered_set<int>& visited, unordered_set<int>& component) const {
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
    eta = std::min(sq, sD);

    // Beta: set based on the frequency of substructures with fewer than 10 edges in the database
    int substructureCount = 0;
    int totalGraphs = 0;
}

class Index {
private:
    Graph tree;

    struct NodeTuple {
        std::string edgeLabel; // Le
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
    std::map<int, NodeTuple> nodeTuples;

    // Method to initialize the mapping of nodes to NodeTuples
    void initializeNodeTuples() {
        // Calculate tree center(s)
        auto centers = tree.calculateTreeCenter();
        if (centers.empty()) {
            throw std::runtime_error("Tree is empty or disconnected.");
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
        std::cout << "Node Tuples:\n";
        for (const auto& [nodeId, tuple] : nodeTuples) {
            std::cout << "Node " << nodeId << ", Label (" << tuple.edgeLabel << ", " << tuple.nodeLabel << ")\n";
        }
    }

    // Method to get and display the canonical form
    void displayCanonicalForm() const {
        std::string canonicalForm = constructCanonicalForm();
        std::cout << "Canonical Form: " << canonicalForm << std::endl;
    }

    // Method to display basic information about the tree (for debugging)
    void displayTreeInfo() const {
        std::cout << "Tree Encoding: " << tree.encode() << std::endl;
        std::cout << "Tree Size (edges): " << tree.size() << std::endl;

        auto centers = tree.calculateTreeCenter();
        std::cout << "Tree Centers: ";
        for (int center : centers) {
            std::cout << center << " (Name: " << tree.vertexNames.at(center) << ") ";
        }
        std::cout << std::endl;
    }

    // Construct the canonical form of the tree using BFS
    std::string constructCanonicalForm() const {
        auto centers = tree.calculateTreeCenter();
        if (centers.empty()) {
            return ""; // Handle empty or disconnected trees
        }
        int root = centers[0];

        // BFS traversal
        std::queue<std::pair<int, int>> bfsQueue; // Pair of (node, parent)
        bfsQueue.push({ root, -1 });

        std::stringstream canonicalForm;

        while (!bfsQueue.empty()) {
            int current = bfsQueue.front().first;
            int parent = bfsQueue.front().second;
            bfsQueue.pop();

            // Add the current node's tuple (Le, Lv, parent Lv) to the canonical form
            const NodeTuple& tuple = nodeTuples.at(current);
            canonicalForm << "(" << tuple.edgeLabel << "," << tuple.nodeLabel << ")";

            // Gather and sort children
            std::vector<NodeTuple> children;
            for (const auto& neighbor : tree.adjList.at(current)) {
                if (neighbor.first != parent) { // Skip the parent node
                    children.push_back({ neighbor.second, tree.vertexNames.at(neighbor.first), neighbor.first });
                }
            }
            std::sort(children.begin(), children.end());

            // Add children to the BFS queue in sorted order
            for (const auto& child : children) {
                bfsQueue.push({ child.nodeId, current });
            }
        }

        return canonicalForm.str();
    }

    // Add to Index class
    bool operator<(const Index& other) const {
        return constructCanonicalForm() < other.constructCanonicalForm();
    }

    // Optionally, add other comparison operators for flexibility
    bool operator>(const Index& other) const {
        return constructCanonicalForm() > other.constructCanonicalForm();
    }

    bool operator==(const Index& other) const {
        return constructCanonicalForm() == other.constructCanonicalForm();
    }

    bool operator!=(const Index& other) const {
        return !(*this == other);
    }

    friend std::ostream& operator<<(std::ostream& os, const Index& index) {
        os << index.constructCanonicalForm() << "\n";
        return os; // Return the stream to allow chaining
    }

};

// B plus tree class
template <typename T> class BPlusTree {
public:
    // structure to create a node
    struct Node {
        bool isLeaf;
        vector<T> keys;
        vector<Node*> children;
        Node* next;

        Node(bool leaf = false)
            : isLeaf(leaf)
            , next(nullptr)
        {
        }
    };

    Node* root;
    // Minimum degree (defines the range for the number of
    // keys)
    int t;

    // Function to split a child node
    void splitChild(Node* parent, int index, Node* child);

    // Function to insert a key in a non-full node
    void insertNonFull(Node* node, T key);

    // Function to remove a key from a node
    void remove(Node* node, T key);

    // Function to borrow a key from the previous sibling
    void borrowFromPrev(Node* node, int index);

    // Function to borrow a key from the next sibling
    void borrowFromNext(Node* node, int index);

    // Function to merge two nodes
    void merge(Node* node, int index);

    // Function to print the tree
    void printTree(Node* node, int level);

public:
    BPlusTree(int degree) : root(nullptr), t(degree) {}

    void insert(T key);
    bool search(T key);
    void remove(T key);
    vector<T> rangeQuery(T lower, T upper);
    void printTree();
};

// Implementation of splitChild function
template <typename T>
void BPlusTree<T>::splitChild(Node* parent, int index,
    Node* child)
{
    Node* newChild = new Node(child->isLeaf);
    parent->children.insert(
        parent->children.begin() + index + 1, newChild);
    parent->keys.insert(parent->keys.begin() + index,
        child->keys[t - 1]);

    newChild->keys.assign(child->keys.begin() + t,
        child->keys.end());
    child->keys.resize(t - 1);

    if (!child->isLeaf) {
        newChild->children.assign(child->children.begin()
            + t,
            child->children.end());
        child->children.resize(t);
    }

    if (child->isLeaf) {
        newChild->next = child->next;
        child->next = newChild;
    }
}

// Implementation of insertNonFull function
template <typename T>
void BPlusTree<T>::insertNonFull(Node* node, T key)
{
    if (node->isLeaf) {
        node->keys.insert(upper_bound(node->keys.begin(),
            node->keys.end(),
            key),
            key);
    }
    else {
        int i = node->keys.size() - 1;
        while (i >= 0 && key < node->keys[i]) {
            i--;
        }
        i++;
        if (node->children[i]->keys.size() == 2 * t - 1) {
            splitChild(node, i, node->children[i]);
            if (key > node->keys[i]) {
                i++;
            }
        }
        insertNonFull(node->children[i], key);
    }
}

// Implementation of remove function
template <typename T>
void BPlusTree<T>::remove(Node* node, T key)
{
    // If node is a leaf
    if (node->isLeaf) {
        auto it = find(node->keys.begin(), node->keys.end(),
            key);
        if (it != node->keys.end()) {
            node->keys.erase(it);
        }
    }
    else {
        int idx = lower_bound(node->keys.begin(),
            node->keys.end(), key)
            - node->keys.begin();
        if (idx < node->keys.size()
            && node->keys[idx] == key) {
            if (node->children[idx]->keys.size() >= t) {
                Node* predNode = node->children[idx];
                while (!predNode->isLeaf) {
                    predNode = predNode->children.back();
                }
                T pred = predNode->keys.back();
                node->keys[idx] = pred;
                remove(node->children[idx], pred);
            }
            else if (node->children[idx + 1]->keys.size()
                >= t) {
                Node* succNode = node->children[idx + 1];
                while (!succNode->isLeaf) {
                    succNode = succNode->children.front();
                }
                T succ = succNode->keys.front();
                node->keys[idx] = succ;
                remove(node->children[idx + 1], succ);
            }
            else {
                merge(node, idx);
                remove(node->children[idx], key);
            }
        }
        else {
            if (node->children[idx]->keys.size() < t) {
                if (idx > 0
                    && node->children[idx - 1]->keys.size()
                    >= t) {
                    borrowFromPrev(node, idx);
                }
                else if (idx < node->children.size() - 1
                    && node->children[idx + 1]
                    ->keys.size()
                    >= t) {
                    borrowFromNext(node, idx);
                }
                else {
                    if (idx < node->children.size() - 1) {
                        merge(node, idx);
                    }
                    else {
                        merge(node, idx - 1);
                    }
                }
            }
            remove(node->children[idx], key);
        }
    }
}

// Implementation of borrowFromPrev function
template <typename T>
void BPlusTree<T>::borrowFromPrev(Node* node, int index)
{
    Node* child = node->children[index];
    Node* sibling = node->children[index - 1];

    child->keys.insert(child->keys.begin(),
        node->keys[index - 1]);
    node->keys[index - 1] = sibling->keys.back();
    sibling->keys.pop_back();

    if (!child->isLeaf) {
        child->children.insert(child->children.begin(),
            sibling->children.back());
        sibling->children.pop_back();
    }
}

// Implementation of borrowFromNext function
template <typename T>
void BPlusTree<T>::borrowFromNext(Node* node, int index)
{
    Node* child = node->children[index];
    Node* sibling = node->children[index + 1];

    child->keys.push_back(node->keys[index]);
    node->keys[index] = sibling->keys.front();
    sibling->keys.erase(sibling->keys.begin());

    if (!child->isLeaf) {
        child->children.push_back(
            sibling->children.front());
        sibling->children.erase(sibling->children.begin());
    }
}

// Implementation of merge function
template <typename T>
void BPlusTree<T>::merge(Node* node, int index)
{
    Node* child = node->children[index];
    Node* sibling = node->children[index + 1];

    child->keys.push_back(node->keys[index]);
    child->keys.insert(child->keys.end(),
        sibling->keys.begin(),
        sibling->keys.end());
    if (!child->isLeaf) {
        child->children.insert(child->children.end(),
            sibling->children.begin(),
            sibling->children.end());
    }

    node->keys.erase(node->keys.begin() + index);
    node->children.erase(node->children.begin() + index
        + 1);

    delete sibling;
}

// Implementation of printTree function
template <typename T>
void BPlusTree<T>::printTree(Node* node, int level)
{
    if (node != nullptr) {
        for (int i = 0; i < level; ++i) {
            cout << "  ";
        }
        for (const T& key : node->keys) {
            cout << key << " ";
        }
        cout << endl;
        for (Node* child : node->children) {
            printTree(child, level + 1);
        }
    }
}

// Implementation of printTree wrapper function
template <typename T> void BPlusTree<T>::printTree()
{
    printTree(root, 0);
}

// Implementation of search function
template <typename T> bool BPlusTree<T>::search(T key)
{
    Node* current = root;
    while (current != nullptr) {
        int i = 0;
        while (i < current->keys.size()
            && key > current->keys[i]) {
            i++;
        }
        if (i < current->keys.size()
            && key == current->keys[i]) {
            return true;
        }
        if (current->isLeaf) {
            return false;
        }
        current = current->children[i];
    }
    return false;
}

// Implementation of range query function
template <typename T>
vector<T> BPlusTree<T>::rangeQuery(T lower, T upper)
{
    vector<T> result;
    Node* current = root;
    while (!current->isLeaf) {
        int i = 0;
        while (i < current->keys.size()
            && lower > current->keys[i]) {
            i++;
        }
        current = current->children[i];
    }
    while (current != nullptr) {
        for (const T& key : current->keys) {
            if (key >= lower && key <= upper) {
                result.push_back(key);
            }
            if (key > upper) {
                return result;
            }
        }
        current = current->next;
    }
    return result;
}

// Implementation of insert function
template <typename T> void BPlusTree<T>::insert(T key)
{
    if (root == nullptr) {
        root = new Node(true);
        root->keys.push_back(key);
    }
    else {
        if (root->keys.size() == 2 * t - 1) {
            Node* newRoot = new Node();
            newRoot->children.push_back(root);
            splitChild(newRoot, 0, root);
            root = newRoot;
        }
        insertNonFull(root, key);
    }
}

// Implementation of remove function
template <typename T> void BPlusTree<T>::remove(T key)
{
    if (root == nullptr) {
        return;
    }
    remove(root, key);
    if (root->keys.empty() && !root->isLeaf) {
        Node* tmp = root;
        root = root->children[0];
        delete tmp;
    }
}

// #TODO
// 1. Querying

int main() {
    vector<Graph> database = setupGraphs("graph.txt"); // Setup the graphs

    // Calculate the average size of query graphs (sq)
    int sq = 6; // Example value for sq (you can adjust this)

    // Calculate alpha, beta, eta
    int alpha, beta = 1, eta;
    calculateAlphaBetaEta(sq, database, alpha, eta);

    unordered_map<Graph, unordered_set<int>, GraphHasher> subtreeFrequency = calculateSubtreeFrequency(database); // Calculate subtree frequencies

    vector<Graph> freqTrees = filterTreesBySupport(subtreeFrequency, alpha, beta, eta); // Filter trees based on support function

    double gamma = 1;
    vector<Graph> finalTrees = shrinkTrees(freqTrees, subtreeFrequency, gamma); // Shrink the trees based on intersection

    vector<Index> indexes;

    BPlusTree<Index> BTree(3);

    for (auto tree : finalTrees)
    {
        Index idx(tree);
        indexes.push_back(idx);
        BTree.insert(idx);
    }

    //BTree.printTree();

    for(auto idx : indexes)
        cout << BTree.search(idx);

    return 0;
}