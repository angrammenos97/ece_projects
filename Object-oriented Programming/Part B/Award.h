#ifndef AWARD_H_INCLUDED
#define AWARD_H_INCLUDED

#include <string>
#include <iostream>

using namespace std;

class Award{
protected:
    string name;
    bool solo;

public:

    Award(){name = "";solo = false;}
    Award(string n, bool s){name = n;solo = s;}

    void setName(string s){name = s;}
    void setSolo(bool b){solo = b;}

    string getName(){return name;}
    bool getSolo(){return solo;}

    void status(){cout << "Name: " << name << endl; cout << "Solo: "; if (solo) cout << "true" << endl; else cout << "false" << endl;}
};

#endif // AWARD_H_INCLUDED
