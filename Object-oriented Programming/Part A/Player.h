#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED
// TODO сулпкгяысте йатаккгка том йыдийа тоу HEADER.
using namespace std;

class Player
{

    int age;
    int wins;
    int team;           // 0 -> μαχητες, 1 -> διασημοι
    string name;
    string gender;
    string profession;
    string item;
    float hunger;
    float strength;

public:

    Player();
    Player(int age, int team, string name, string gender, string profession, string item);
    ~Player();
    void setAge(int a);
    void setWins(int a);
    void setName(string a);
    void setGender(string a);
    void setProfession(string a);
    void setItem(string a);
    void setHunger(float a);
    void setStrength(float a);
    void setTeam(int a);
    int getAge();
    int getWins();
    string getName();
    string getGender();
    string getProfession();
    string getItem();
    float getHunger();
    float getStrength();
    int getTeam();
    void eat();
    void work();
    void sleep();
};

#endif // PLAYER_H_INCLUDED
