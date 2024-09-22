#ifndef FOODAWARD_H_INCLUDED
#define FOODAWARD_H_INCLUDED

#include "Award.h"
#include <cstdlib>

using namespace std;

class FoodAward : public Award{

    int bonusSupplies;

public:

    FoodAward(){bonusSupplies = 0;}
    FoodAward(string n, bool s) : Award(n,s) {bonusSupplies = 10 + rand() % 30;}

    void setBonusSupplies(int val){bonusSupplies = val;}
    int getBonusSupplies(){return bonusSupplies;}

    void status(){Award::status(); cout << "Supplies: " << bonusSupplies << endl << endl; }

};


#endif // FOODAWARD_H_INCLUDED
