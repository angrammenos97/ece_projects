#ifndef TEAM_H_INCLUDED
#define TEAM_H_INCLUDED

using namespace std;

class Team
{

    string name;
    int playerNum;
    int food;
    int wins;

public:
    Team();
    Team(string a);
    ~Team();
    void setName(string a);
    void setPlayerNum(int a);
    void setFood(int a);
    void setWins(int a);
    string getName();
    int getPlayerNum();
    int getFood();
    int getWins();
};

#endif // TEAM_H_INCLUDED
