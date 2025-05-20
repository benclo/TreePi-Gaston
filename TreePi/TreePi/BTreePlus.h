#ifndef BPLUSTREE_H
#define BPLUSTREE_H

#include <vector>
#include <iostream>
#include <algorithm>

template <typename T>
class BPlusTree {
public:
    struct Node {
        bool isLeaf;
        std::vector<T> keys;
        std::vector<Node*> children;
        Node* next;

        Node(bool leaf = false) : isLeaf(leaf), next(nullptr) {}
    };

    BPlusTree(int degree);
    void insert(T key);
    bool search(T key);
    void printTree();
    int splitCount = 0;
    int getSplitCount() const { return splitCount; }

private:
    Node* root;
    int t; // Minimum degree

    void splitChild(Node* parent, int index, Node* child);
    void insertNonFull(Node* node, T key);
    void printTree(Node* node, int level);
};

#endif // BPLUSTREE_H
