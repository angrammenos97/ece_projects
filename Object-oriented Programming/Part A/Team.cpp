// TODO сулпкгяысте йатаккгка том йыдийа тгс йкасгс TEAM.
#include <iostream>
#include <string>
#include "Team.h"

Team::Team(string a)
{
    name = a;
    playerNum = 0;
    food = 0;
    wins = 0;
}

// Getters - Setters
Team::~Team()
{
    cout << "Team " << name << " destroyed!" << endl;
}
void Team::setName(string a)
{
    name = a;
}
void Team::setPlayerNum(int a)
{
    playerNum = a;
}
void Team::setFood(int a)
{
    food = a;
}
void Team::setWins(int a)
{
    wins = a;
}
string Team::getName()
{
    return name;
}
int Team::getPlayerNum()
{
    return playerNum;
}
int Team::getFood()
{
    return food;
}
int Team::getWins()
{
    return wins;
}
