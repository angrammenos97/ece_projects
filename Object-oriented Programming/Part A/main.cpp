#include <iostream>
#include <string>
#include <cstdio>
#include <stdlib.h> //to clear terminal
#include <conio.h>  // to use getch()
#include "Team.h"
#include "Player.h"


using namespace std;

Team m("Maxhtes");
Team d("Diashmoi");
Player playerM[10], playerD[10];

void addPlayer();
void teamStats();
void showPlayer();

int main()
{
    char choice = 0;

    system("cls||@clear");


    //start of loop
    while(choice != 4)
    {
        cout << "*Hello SURVIVORS!*\n" << endl;
        //cout << "This is a simulator of critically acclaimed TV Reality-Show SURVIVOR."
        cout << "Choose an option: (1, 2, 3, 4)" << endl;
        cout << " 1: Add Player..." << endl;
        cout << " 2: Show Team Stats..." << endl;
        cout << " 3: Show Player Attributes..." << endl;
        cout << " 4: Exit Game... ( why would you exit?! )" << endl;

        cout << "Enter your choice: ";
        cin >> choice;

        switch(choice)
        {

        case 49:        // the choice 1
            cin.clear();       //clear cin buffer
            fflush(stdin);    //clear cin buffer
            addPlayer();
            break;
        case 50:        // the choice 2
            cin.clear();       //clear cin buffer
            fflush(stdin);    //clear cin buffer
            teamStats();
            break;
        case 51:        // the choice 3
            cin.clear();       //clear cin buffer
            fflush(stdin);    //clear cin buffer
            showPlayer();
            break;
        case 52:        // the choice 4
            cout << "\nPress any key to continue.."<< endl;
            getch();
            return 0;
        default :
            cout << "\nWrong input.Please try again.."<< endl;
            cin.clear();       //clear cin buffer
            fflush(stdin);    //clear cin buffer
            cout << "Press any key to continue.."<< endl;
            getch();
        }

        cin.clear();       //clear cin buffer
        fflush(stdin);    //clear cin buffer
        system("cls||@clear");
    }
    return 0;
}

void addPlayer()
{
    char teamChoice = 0;
    // attribute temporary variables
    string tempS=" ";
    int tempI = 0;
    //float tempF = 0.0;

    cout << "\nSelect team( 1:Maxhtes, 2:Diashmoi ): " ;
    cin >> teamChoice;
    if(teamChoice == 49)    //choice 1
    {
        cin.clear();       //clear cin buffer
        fflush(stdin);    //clear cin buffer

        // check if team is full
        if (m.getPlayerNum() == 10)
        {
            cout << "This team is full!" << endl;
            cout << "\nPress any key to continue.."<< endl;
            getch();
            return;
        }
        else
        {
            cout << "\nSelect name:\t";
            cin >> tempS;
            playerM[m.getPlayerNum()].setName(tempS);
            //cout << playerM[m.getPlayerNum()].getName()<< endl;
            cout << "Select gender:\t";
            cin >> tempS;
            // add check
            playerM[m.getPlayerNum()].setGender(tempS);
            cout << "Select profession:\t";
            cin >> tempS;
            playerM[m.getPlayerNum()].setProfession(tempS);
            cout << "Select an item for the player to bring with him/her:\t";
            cin >> tempS;
            playerM[m.getPlayerNum()].setItem(tempS);
            cout << "Select age:\t";
            cin >> tempI;
            playerM[m.getPlayerNum()].setAge(tempI);
            playerM[m.getPlayerNum()].setTeam(0);
            m.setPlayerNum(m.getPlayerNum() + 1); // increase player counter
            m.setFood(m.getFood() + 30); // allocate food portions to team
            cout << endl << "Team Maxhtes now has " << m.getPlayerNum() << " player(s)!" << endl ;

            cout << "\nPress any key to continue.."<< endl;
            getch();
        }
    }
    else if(teamChoice == 50)   //choice 2
    {
        cin.clear();       //clear cin buffer
        fflush(stdin);    //clear cin buffer

        // check if team is full
        if (d.getPlayerNum() == 10)
        {
            cout << "This team is full!" << endl;
            cout << "\nPress any key to continue.."<< endl;
            getch();
            return;
        }
        else
        {
            cout << "\nSelect name:\t";
            cin >> tempS;
            playerD[d.getPlayerNum()].setName(tempS);
            //cout << playerM[d.getPlayerNum()].getName() << endl;
            cout << "Select gender:\t";
            cin >> tempS;
            // add check
            playerD[d.getPlayerNum()].setGender(tempS);
            cout << "Select profession:\t";
            cin >> tempS;
            playerD[d.getPlayerNum()].setProfession(tempS);
            cout << "Select an item for the player to bring with him/her:\t";
            cin >> tempS;
            playerD[d.getPlayerNum()].setItem(tempS);
            cout << "Select age:\t";
            cin >> tempI;
            playerD[d.getPlayerNum()].setAge(tempI);
            playerD[d.getPlayerNum()].setTeam(1);
            d.setPlayerNum(d.getPlayerNum() + 1); // increase player counter
            d.setFood(d.getFood() + 30); // allocate food portions to team
            cout << endl << "Team Diashmoi now has " << d.getPlayerNum() << " player(s)!" << endl;

            cout << "\nPress any key to continue.."<< endl;
            getch();
        }
    }
    else
    {
        cout << "\nWrong input. Please try again"<< endl;
        cin.clear();       //clear cin buffer
        fflush(stdin);    //clear cin buffer
        cout << "Press any key to continue.."<< endl;
        getch();
        addPlayer();
    }
}

void teamStats()
{
    system("cls||@clear");
    cout << ">Team Maxhtes:" << endl;
    cout << " Players:\t" << m.getPlayerNum() << endl;
    cout << " Wins:\t" << m.getWins() << endl;
    cout << " Food remaining:\t" << m.getFood() << " portions" << endl;
    cout << endl << ">Team Diashmoi:" << endl;
    cout << " Players:\t" << d.getPlayerNum() << endl;
    cout << " Wins:\t" << d.getWins() << endl;
    cout << " Food remaining:\t" << d.getFood() << " portions" << endl;
    cout << "\nPress any key to continue.."<< endl;
    getch();
}

void showPlayer()
{
    string playerChoice=" ";
    bool found=0;   // to show or not the messege "nothing found"
    cout << "\nInput Player Name:\t";
    cin >> playerChoice;
    for(int i = 0; i < m.getPlayerNum() ; i++)
    {
        if(playerChoice == playerM[i].getName())
        {
            cout << endl << playerM[i].getName() << " :" << endl;
            cout << " Team:\tMaxhtes" << endl;
            cout << " Gender:\t" << playerM[i].getGender() << endl;
            cout << " Age:\t" << playerM[i].getAge() << endl;
            cout << " Profession:\t" << playerM[i].getProfession() << endl;
            cout << " Personal Item:\t" << playerM[i].getItem() << endl << endl;
            cout << " Personal Wins:\t" << playerM[i].getWins() << endl;
            cout << " Strength:\t" << playerM[i].getStrength() << "%" << endl;
            cout << " Hunger:\t" << playerM[i].getHunger() << "%" << endl;

            found=1;
            cout << "\nPress any key to continue.."<< endl;
            getch();
        }
    }

    for(int i = 0; i < d.getPlayerNum() ; i++)
    {
        if(playerChoice == playerD[i].getName())
        {
            cout << endl << playerD[i].getName() << " :" << endl;
            cout << " Team:\tDiashmoi" << endl;
            cout << " Gender:\t" << playerD[i].getGender() << endl;
            cout << " Age:\t" << playerD[i].getAge() << endl;
            cout << " Profession:\t" << playerD[i].getProfession() << endl;
            cout << " Personal Item:\t" << playerD[i].getItem() << endl << endl;
            cout << " Personal Wins:\t" << playerD[i].getWins() << endl;
            cout << " Strength:\t" << playerD[i].getStrength() << "%" << endl;
            cout << " Hunger:\t" << playerD[i].getHunger() << "%" << endl;

            found=1;
            cout << "\nPress any key to continue.."<< endl;
            getch();
        }
    }

    if (!found)
    {
        cout <<"Nothing found with the name " << playerChoice <<endl;
        cout << "\nPress any key to continue.."<< endl;
        getch();
    }
}
