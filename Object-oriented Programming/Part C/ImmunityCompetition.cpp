#include "ImmunityCompetition.h"

using namespace std;

void ImmunityCompetition::compete(Team &team, ImmunityAward immunityAward)
{
    string winner = "";
    int maxIndex = -1;
    float maxValue = -1;

    for(int i = 0; i < 10; i++)
    {
        if (maxValue < team.getPlayers()[i].getPower() && team.getPlayers()[i].getName() != "")
        {
            maxValue = team.getPlayers()[i].getPower();
            maxIndex = i;
        }

    }

    winner = team.getPlayers()[maxIndex].getName();
    team.getPlayers()[maxIndex].setImmunity(true);
    team.getPlayers()[maxIndex].setVotes(1+immunityAward.getVotes());

    cout << "Immunity Competition Finished." << endl;
    cout << "Winner is " << winner << "." << endl;
    team.getPlayers()[maxIndex].status();
}
