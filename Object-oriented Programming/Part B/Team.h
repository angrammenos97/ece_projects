#ifndef TEAM_H_INCLUDED
#define TEAM_H_INCLUDED

#include <iostream>
#include <string>
#include "Player.h"

using namespace std;

class Team{

    string name;
    int wins;
    int supplies;
    Player *players;

public:

    Team(){name = ""; wins = 0; supplies = 0; players = new Player[10];}
    Team(string name);
    ~Team(){delete [] players; cout << "Team " << name << " is destroyed." << endl;}

    string getName(){return name;}
    int getWins(){return wins;}
    int getSupplies(){return supplies;}
    int getNumberOfPlayers();
    Player* getPlayers(){return players;}

    void setName(string val){name = val;}
    void setWins(int val){wins = val;}
    void setSupplies(int val){supplies = val;}

    void teamWorks();
    void teamEats();
    void teamSleeps();
    void status(bool playerInformation);
};

#endif // TEAM_H_INCLUDED
