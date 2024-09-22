#include "Team.h"
#include "Player.h"

using namespace std;

Team::Team(string s)
{
    srand (time(NULL));
    name = s;

    if (s == "Diasimoi")
        players = new Player[10] {Player("Spal", male, 30), Player("Bo", male, 32),Player("Laoura", female, 28),Player("Choutos", male, 38),Player("Chantavlakas", male, 27),Player("Kokkinakis", male, 40),Player("Chraniolas", male, 38),Player("Evridiki", female, 25), Player("Eirini (Pap)", female, 27),Player("Danos", male, 26)};
    else
        players = new Player[10] {Player("Panaras", male, 37), Player("Kostas(Merc)", male, 33), Player("Elissavet", female, 24), Player("Konstantinos(Modelo)", male, 22), Player("Marios", male, 29), Player("Tsangk", male, 24), Player("Willy", male, 24),Player("Sara", female, 24), Player("Eirini (Ain)", female, 29),Player("Anergos", male, 25)};

}

int Team::getNumberOfPlayers()
{
    int result = 0;

    for (int i = 0; i < 10; i++)
        if (players[i].getAge() != 0)
            result++;

    return result;
}

void Team::status(bool playerInformation)
{

    cout << "Name: " << name << endl;

    if (playerInformation)
    {
        cout << endl << "Players:" << endl;
        for (int i = 0; i < 10; i++)
            if (players[i].getAge() != 0)
                players[i].status();
    }
    cout << endl << endl;
}

void Team::setCandidate(string player){

    for (int i = 0; i < 10; i++){

        if (player == players[i].getName()){
            players[i].setCandidate(true);
            break;
        }

    }

    //status(true);

}

void Team::removePlayer(string player){

    Player p;

    for (int i = 0; i < 10; i++){

        if (player == players[i].getName()){
            players[i] = p;
            break;
        }

    }

    //status(true);

}
