#include "Team.h"
//#include "Competition.h"
#include "TeamCompetition.h"
#include "QuizCompetition.h"
#include "ImmunityCompetition.h"

using namespace std;

Team teams[2] = { Team("Diasimoi"), Team("Maxites")};
int competitionId = 0;
//void testAwards();
void menu();
void normalDay();
void teamCompetitionDay();
void quizDay();

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
        cout << "1.Normal Day." << endl;
        cout << "2.Team Competition Day." << endl;
        cout << "3.Quiz Day." << endl;
        cout << "0.Quit" << endl;

        cin >> choice;

        switch(choice)
        {

        case 1:
            normalDay();
            break;
        case 2:
            teamCompetitionDay();
            break;
        case 3:
            quizDay();
            break;
        case 0:
            break;
        default:
            cout << "Incorrect Input. Choose between 1 and 3. Press 0 to quit." << endl;

        }
    }
}

void normalDay()
{

    cout << "This is a normal day in the Survivor Game." << endl << endl;

    for (unsigned int i=0; i<2; i++)
    {
        teams[i].teamWorks();
        teams[i].teamEats();
        teams[i].teamSleeps();
        cout << "Team " << teams[i].getName() << " went through their routine..." << endl;
    }
    cout << endl;

}

void teamCompetitionDay()
{
    cout << "This is a team competition day in the Survivor Game." << endl << endl;

    for (unsigned int i=0; i<2; i++)
    {
        teams[i].teamWorks();
    }
    cout << "Insert today's food award: ";
    string tempS;
    cin >> tempS;
    FoodAward todaysFoodAward(tempS,0);
    todaysFoodAward.status();
    cout << "Insert Competition's name: ";
    cin >> tempS;
    TeamCompetition todaysTeamCompetition(competitionId + 1,tempS,todaysFoodAward);
    competitionId++;
    cout << "Here are the results!" << endl;
    if(todaysTeamCompetition.compete(teams[0],teams[1]))
    {
        cout << "Team Diashmoi have won!" << endl;
        cout << "Team Maxhtes now have to face an Immunity Competition!" << endl;
        cout << "Insert Immunity Award's name: ";
        cin >> tempS;
        ImmunityAward todaysImmunityAward(tempS, 1);
        todaysImmunityAward.status();
        cout << "Insert Competition's name: ";
        cin >> tempS;
        cout << "...the results are:" << endl;
        ImmunityCompetition todaysImmunityCompetition(competitionId + 1, tempS, todaysImmunityAward);
        todaysImmunityCompetition.compete(teams[1]);
    }
    else
    {
        cout << "Team Maxhtes have won!" << endl;
        cout << "Team Diashmoi now have to face an Immunity Competition!" << endl;
        cout << "Insert Immunity Award's name: ";
        cin >> tempS;
        ImmunityAward todaysImmunityAward(tempS, 1);
        todaysImmunityAward.status();
        cout << "Insert Competition's name: ";
        cin >> tempS;
        cout << "...the results are:" << endl;
        ImmunityCompetition todaysImmunityCompetition(competitionId + 1, tempS, todaysImmunityAward);
        todaysImmunityCompetition.compete(teams[0]);
    }
    for (unsigned int i=0; i<2; i++)
    {
        teams[i].teamEats();
        teams[i].teamSleeps();
        cout << "Team " << teams[i].getName() << " went through their routine..." << endl;
    }
    cout << endl;

}

void quizDay()
{

    cout << "This is a quiz day in the Survivor Game." << endl << endl;

    for (unsigned int i=0; i<2; i++)
    {
        teams[i].teamWorks();
    }
    cout << "Insert today's Communication award: ";
    string tempS;
    cin >> tempS;
    CommunicationAward todaysCommunicationAward(tempS,0);
    todaysCommunicationAward.status();
    cout << "Insert Competition's name: ";
    cin >> tempS;
    QuizCompetition todaysQuizCompetition(competitionId + 1,tempS,todaysCommunicationAward);
    competitionId++;
    cout << "Here are the results!" << endl;
    todaysQuizCompetition.compete(teams[0], teams[1]);
    for (unsigned int i=0; i<2; i++)
    {
        teams[i].teamEats();
        teams[i].teamSleeps();
        cout << "Team " << teams[i].getName() << " went through their routine..." << endl;
    }
    cout << endl;


}
