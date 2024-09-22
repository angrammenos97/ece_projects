#ifndef TEAM_H_INCLUDED
#define TEAM_H_INCLUDED

#include <iostream>
#include <cstdlib>
#include <time.h>
#include <string>
#include "Player.h"

using namespace std;

class Team{

    string name;
    Player *players;

public:

    Team(){name = ""; players = new Player[10];}
    Team(string name);
    ~Team(){delete [] players; cout << "Team " << name << " is destroyed." << endl;}

    string getName(){return name;}
    Player* getPlayers(){return players;}

    void setName(string val){name = val;}

    int getNumberOfPlayers();
    void setCandidate(string player);
    void removePlayer(string player);

    void status(bool playerInformation);

};

#endif // TEAM_H_INCLUDED
