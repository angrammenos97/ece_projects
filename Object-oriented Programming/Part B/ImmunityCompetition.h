#ifndef IMMUNITYCOMPETITION_H_INCLUDED
#define IMMUNITYCOMPETITION_H_INCLUDED
#include "Competition.h"
#include "ImmunityAward.h"

using namespace std;

class ImmunityCompetition:public Competition
{

    ImmunityAward immunityAward;
public:
    ImmunityCompetition() {};
    ImmunityCompetition(int id,string name,ImmunityAward immunityAward):Competition(id,name)
    {
        this -> id = id;
        this -> name = name;
        this -> immunityAward = immunityAward;
    }
    ~ImmunityCompetition()
    {
        cout << "Immunity Competition " << id << " destroyed!" << endl;
    }
    //status
    void status()
    {
        cout << "This is an Immunity Competition." << endl;
        Competition::status();
    }

    void compete(Team &team)
    {
        float tempPower = 0;
        Player tempPlayer;
        for (int i = 0; i<team.getNumberOfPlayers(); i++)
        {
            tempPlayer = *(team.getPlayers() + i);
            cout << tempPlayer.getName() << ":  " << tempPlayer.getPower() << endl;
            if(tempPlayer.getPower() > tempPower)
            {
                tempPower = tempPlayer.getPower();
                setWinner(tempPlayer.getName());
            }
        }
        cout << getWinner() << " wins the Immunity Award!" << endl;
    }
};


#endif // IMMUNITYCOMPETITION_H_INCLUDED
