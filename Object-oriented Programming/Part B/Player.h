#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>

using namespace std;

enum genderOption {male, female};

class Player{

    string name;
    genderOption gender;
    int age;
    float power;
    float hunger;
    bool candidate;

public:
    Player();
    Player(string n, genderOption g, int a);
    ~Player();

    string getName();
    int getAge();
    genderOption getGender();
    float getPower();
    float getHunger();
    bool getCandidate();

    void setName(string val);
    void setPower(float val);
    void setHunger(float val);
    void setCandidate(bool val);

    void work();
    void eat();
    void sleep();
    void compete();

    void status();
};


#endif // PLAYER_H_INCLUDED
