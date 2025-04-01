// main.cpp
// Siegfried Nijssen, snijssen@liacs.nl, jan 2004.
#include <iostream>
#include <fstream>
#include <unordered_map>
#include "Graph.h"
#include "database.h"
#include "path.h"
#include "misc.h"
#include "graphstate.h"
#include "gaston.h"

#include <time.h>
#define _CRT_SECURE_NO_WARNINGS


#ifdef _WIN32
#include "C:\Users\matas\Desktop\Universitetas\Ketvirtas kursas\Astuntas semestras\Bakalauras\getopt\getopt_port\getopt.h"
#endif

using namespace std;

Frequency minfreq = 1;
Database database;
Statistics statistics;
bool dooutput = false;
int phase = 3;
int maxsize = ( 1 << ( sizeof(NodeId)*8 ) ) - 1; // safe default for the largest allowed pattern
FILE *output;

void Statistics::print () {
  int total = 0, total2 = 0, total3 = 0;
  for ( int i = 0; i < frequenttreenumbers.size (); i++ ) {
    cout << "Frequent " << i + 2
         << " cyclic graphs: " << frequentgraphnumbers[i]
         << " real trees: " << frequenttreenumbers[i]
         << " paths: " << frequentpathnumbers[i]
         << " total: " << frequentgraphnumbers[i] + frequenttreenumbers[i] + frequentpathnumbers[i] << endl;
    total += frequentgraphnumbers[i];
    total2 += frequenttreenumbers[i];
    total3 += frequentpathnumbers[i];
  }
  cout << "TOTAL:" << endl
       << "Frequent cyclic graphs: " << total << " real trees: " << total2 << " paths: " << total3 << " total: " << total + total2 + total3 << endl;
}

void puti ( FILE *f, int i ) {
  char array[100];
  int k = 0;
  do {
    array[k] = ( i % 10 ) + '0';
    i /= 10;
    k++;
  }
  while ( i != 0 );
  do {
    k--;
    putc ( array[k], f );
  } while ( k );
}

unordered_map<Graph, int, GraphHasher> gaston(int freq, const std::string& inputFile, const std::string& outputFile) {
  clock_t t1 = clock ();
  cerr << "GASTON GrAph, Sequences and Tree ExtractiON algorithm" << endl;
  cerr << "Version 1.0 with Occurrence Lists" << endl;
  cerr << "Siegfried Nijssen, LIACS, 2004" << endl;

  unordered_map<Graph, int, GraphHasher> trees;

  phase = 2;
  cout << inputFile;
  minfreq = freq;
  cerr << "Read" << endl;
  FILE *input = fopen (inputFile.c_str(), "r" );
  dooutput = true;
  output = fopen(outputFile.c_str(), "w");
  database.read ( input );
  fclose ( input );
  cerr << "Edgecount" << endl;
  database.edgecount ();
  cerr << "Reorder" << endl;
  database.reorder ();

  initLegStatics ();
  graphstate.init ();
  for ( int i = 0; i < database.nodelabels.size (); i++ ) {
    if ( database.nodelabels[i].frequency >= minfreq &&
         database.nodelabels[i].frequentedgelabels.size () ) {
      Path path ( i );
      path.expand (trees);
    }
  }

  clock_t t2 = clock ();

  statistics.print ();
  cout << "Approximate total runtime: " << ( (float) t2 - t1 ) / CLOCKS_PER_SEC << "s" << endl;
  fclose(output);

  return trees;
}
