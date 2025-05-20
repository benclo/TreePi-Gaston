#include "BTreePlus.h"

template <typename T>
BPlusTree<T>::BPlusTree(int degree) : root(nullptr), t(degree) {}

template <typename T>
void BPlusTree<T>::splitChild(Node* parent, int index, Node* child) {
    splitCount++; // <-- Count every time a split happens

    Node* newChild = new Node(child->isLeaf);
    parent->children.insert(parent->children.begin() + index + 1, newChild);
    parent->keys.insert(parent->keys.begin() + index, child->keys[t - 1]);

    newChild->keys.assign(child->keys.begin() + t, child->keys.end());
    child->keys.resize(t - 1);

    if (!child->isLeaf) {
        newChild->children.assign(child->children.begin() + t, child->children.end());
        child->children.resize(t);
    }

    if (child->isLeaf) {
        newChild->next = child->next;
        child->next = newChild;
    }
}


template <typename T>
void BPlusTree<T>::insertNonFull(Node* node, T key) {
    if (node->isLeaf) {
        node->keys.insert(std::upper_bound(node->keys.begin(), node->keys.end(), key), key);
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

template <typename T>
void BPlusTree<T>::insert(T key) {
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

template <typename T>
bool BPlusTree<T>::search(T key) {
    Node* current = root;
    while (current != nullptr) {
        int i = 0;
        while (i < current->keys.size() && key > current->keys[i]) {
            i++;
        }
        if (i < current->keys.size() && key == current->keys[i]) {
            return true;
        }
        if (current->isLeaf) {
            return false;
        }
        current = current->children[i];
    }
    return false;
}

template <typename T>
void BPlusTree<T>::printTree(Node* node, int level) {
    if (node != nullptr) {
        for (int i = 0; i < level; ++i) {
            std::cout << "  ";
        }
        for (const T& key : node->keys) {
            std::cout << key << " ";
        }
        std::cout << std::endl;
        for (Node* child : node->children) {
            printTree(child, level + 1);
        }
    }
}

template <typename T>
void BPlusTree<T>::printTree() {
    printTree(root, 0);
}
