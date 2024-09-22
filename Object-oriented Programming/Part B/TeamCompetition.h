#ifndef TEAMCOMPETITION_H_INCLUDED
#define TEAMCOMPETITION_H_INCLUDED
#include <cstring>
#include <ctime>
#include <stdlib.h>
#include "Competition.h"
#include "FoodAward.h"
#include "Round.h"

using namespace std;

class TeamCompetition: public Competition
{

    FoodAward foodAward;
    Round rounds[19];

public:

    TeamCompetition() {};
    TeamCompetition(int id, string name, FoodAward foodAward):Competition(id, name)
    {
        this-> id = id;
        this-> name = name;
        this-> foodAward = foodAward;
    };
    ~TeamCompetition()
    {
        cout << "Team Competition " << id << " destroyed!" << endl;
    };
    // setters
    // getters
    // status
    void status()
    {
        cout << "This is a Team Competition." << endl;
        Competition::status();
    }

    int compete(Team &team1, Team &team2)
    {
        for(int i = 0; i < 19; i++)
        {
            rounds[i].setDuration(500);
            rounds[i].setId(i+1);
        }
        int wins1 = 0, wins2 = 0;
        Player player1, player2;
        srand(time(NULL));
        while(wins1 < 10 && wins2 < 10)
        {
            player1 = *(team1.getPlayers() + (rand() % team1.getNumberOfPlayers()));
            player2 = *(team2.getPlayers() + (rand() % team2.getNumberOfPlayers()));
            player1.compete();
            player2.compete();
            if(player1.getPower() > player2.getPower())
            {
                rounds[wins1 + wins2].setWinner(player1.getName());
                wins1++;
            }
            else if(player1.getPower() < player2.getPower())
            {
                rounds[wins1 + wins2].setWinner(player2.getName());
                wins2++;
            }
            else if(player1.getHunger() > player2.getHunger())
            {
                rounds[wins1 + wins2].setWinner(player1.getName());
                wins1++;
            }
            else if(player1.getHunger() < player2.getHunger())
            {
                rounds[wins1 + wins2].setWinner(player2.getName());
                wins2++;
            }
        }
        for(int i = 0; i < wins1 + wins2; i++)
        {
            //cout << "Round " << i + 1 << " winner: " << rounds[i].getWinner() << endl;
            rounds[i].status();
        }
        if(wins1 == 10)
        {
            cout << endl << "Final Score:" << endl << "Diashmoi: " << wins1 << endl << "Maxhtes: " << wins2 << endl;
            team1.setWins(team1.getWins() + 1);
            team1.setSupplies(team1.getSupplies() + foodAward.getBonusSupplies());
            return 1;
        }
        else
        {
            cout << endl << "Final Score:" << endl << "Diashmoi: " << wins1 << endl << "Maxhtes: " << wins2 << endl;
            team2.setWins(team2.getWins() + 1);
            team2.setSupplies(team2.getSupplies() + foodAward.getBonusSupplies());
            return 0;
        }
    }
};

#endif // TEAMCOMPETITION_H_INCLUDED
