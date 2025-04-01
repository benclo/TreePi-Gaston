#ifndef GASTON_H
#define GASTON_H

#include <string>
#include <cstdio>
#include "database.h"
#include "misc.h"

#ifdef _WIN32
#include "C:\Users\matas\Desktop\Universitetas\Ketvirtas kursas\Astuntas semestras\Bakalauras\getopt\getopt_port\getopt.h"
#endif

// Declare the global variables
extern int phase;
extern int maxsize;
extern Frequency minfreq;
extern Database database;
extern Statistics statistics;
extern bool dooutput;
extern FILE* output;

// Declare the functions
void puti(FILE* f, int i);
unordered_map<Graph, int, GraphHasher> gaston(int freq, const std::string& inputFile, const std::string& outputFile);

#endif // GASTON_H
