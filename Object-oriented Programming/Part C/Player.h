#ifndef PLAYER_H_INCLUDED
#define PLAYER_H_INCLUDED

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <time.h>

using namespace std;

enum genderOption {male, female};

class Player{

    string name;
    genderOption gender;
    int age;
    float power;
    float hunger;
    bool candidate;
    bool immunity;
    int wins;
    int votes;

public:
    Player(){

        name = "";
        gender = female;
        age = 0;
        power = 100;
        hunger = 0;
        candidate = false;
        immunity = false;
        wins = 0;
        votes = 0;
    }

    Player(string n, genderOption g, int a){

        name = n;
        gender = g;
        age = a;
        candidate = false;
        immunity = false;
        power = rand()%100;
        hunger = rand()%100;
        wins = rand() % 10;
        votes = 1;
    }

    ~Player(){
        //cout << "Player " << name << " is destroyed." << endl;
    }

    string getName(){return name;}
    int getAge(){return age;}
    genderOption getGender(){return gender;}
    float getPower(){return power;}
    float getHunger(){return hunger;}
    bool getCandidate(){return candidate;}
    bool getImmunity(){return immunity;}
    int getWins(){return wins;}
    int getVotes(){return votes;}

    void setName(string val){name = val;}
    void setPower(float val){power = val;}
    void setHunger(float val){hunger = val;}
    void setCandidate(bool val){candidate = val;}
    void setImmunity(bool val){immunity = val;}
    void setWins(int val){wins = val;}
    void setVotes(int val){votes = val;}

    void status(){

        cout << "Name: " << name;
        cout << " Gender: ";
        if (gender == male)
            cout << "Male";
        else
            cout << "Female";
        cout << " Age: " << age;
        cout << " Power: " << power;
        cout << " Hunger: " << hunger;
        cout << " Votes: " << votes;
        cout << " Candidate: ";
        if (candidate == 0)
            cout << "false";
        else
            cout << "true";
        cout << " Immunity: ";
        if (immunity == 0)
            cout << "false";
        else
            cout << "true";
        cout << endl << endl;

    }

};


#endif // PLAYER_H_INCLUDED
