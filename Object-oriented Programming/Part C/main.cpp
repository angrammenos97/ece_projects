#include "Player.h"
#include "Team.h"
#include "ImmunityCompetition.h"
#include "Voting.h"

using namespace std;

Team teams[2] = { Team("Diasimoi"), Team("Maxites")};
void menu();
void votingDay();
void immunityProcess(int teamChoice);
void votingProcess(int teamChoice);

int main()
{
    menu();

    return 0;
}

void menu()
{
    int choice = -1;

    while(choice != 0)
    {
        cout << "1.Voting Day." << endl;
        cout << "0.Quit" << endl;

        cin >> choice;

        switch(choice)
        {

        case 1:
            votingDay();
            break;
        case 0:
            break;
        default:
            cout << "Incorrect Input. Choose between 1  Press 0 to quit." << endl;

        }
    }
}

void votingDay()
{

    cout << "This is voting process in the Survivor Game." << endl << endl;

    int teamChoice = -1;
    while (teamChoice != 0 && teamChoice != 1){
        cout << "Choose Team for immunity competition:0.Diasimoi or 1.Maxites" << endl;
        cin >> teamChoice;
    }

    immunityProcess(teamChoice);

    Voting::votingProcess(teams[teamChoice]);

}

void immunityProcess(int teamChoice)
{

    cout << "This is immunity process in the Survivor Game for " << teams[teamChoice].getName() << "." << endl << endl;

    ImmunityAward immunityAward("Immunity Award");
    ImmunityCompetition::compete(teams[teamChoice], immunityAward);

}

