// TO FILL UP
#include "Voting.h"
#include <cstdlib>
#include <vector>
#include <iostream>

using namespace std;

vector<Vote>Voting::votes;
map<string, int>Voting::results;

void Voting::genderVoting(Team &team, int playerTurn)
{
    string reasonWhy = "Different gender";
    // temp player
    Player player = *(team.getPlayers() + playerTurn);
    // counter for infinity loop
    int tempcounter=0;
    //number of votes per player
    while(player.getVotes())
    {
        // temp voted player
        Player votedPlayer = *(team.getPlayers() + rand()%10);
        // check
        if(votedPlayer.getName() != player.getName() && !votedPlayer.getImmunity() && votedPlayer.getName() != "" && votedPlayer.getGender() != player.getGender())
        {
            Vote tempVote;
            tempVote.setVoted(votedPlayer.getName());
            tempVote.setReason(reasonWhy);
            votes.push_back(tempVote);
            player.setVotes(player.getVotes() - 1);
            //cout << player.getName() << " voted " << votedPlayer.getName() << endl;
        }
        // case infinity loop
        else
        {
            //cout << player.getName() <<  "*Inf loop++*" << votedPlayer.getName() << endl;
            tempcounter++;
        }
        // exit from infinity loop by doing random voting
        if (tempcounter == 100)
        {
            //cout << "*Chaning to random voting*" << endl;
            randomVoting(team, playerTurn);
            return;
        }
    }
}

void Voting::randomVoting(Team &team, int playerTurn)
{
    string reasonWhy = "Random";
    // temp player
    Player player = *(team.getPlayers() + playerTurn);
    //number of votes per player
    while(player.getVotes())
    {
        // temp voted player
        Player votedPlayer = *(team.getPlayers() + rand()%10);
        // check
        if(votedPlayer.getName() != player.getName() && !votedPlayer.getImmunity() && votedPlayer.getName() != "")
        {
            Vote tempVote;
            tempVote.setVoted(votedPlayer.getName());
            tempVote.setReason(reasonWhy);
            votes.push_back(tempVote);
            player.setVotes(player.getVotes() - 1);
            //cout << player.getName() << " voted " << votedPlayer.getName() << endl;
        }
    }
}

void Voting::lessPowerVoting(Team &team, int playerTurn)
{
    string reasonWhy = "He/She has the least amount of Power";
    // find least power
    int leastPower = 100;
    // position of player with least power
    int powerId=-1;
    // temp player
    Player player;

    for(int j = 0; j < 10; j++)
    {
        player = *(team.getPlayers() + j);
        if(player.getPower() < leastPower && player.getPower() > 0 && !player.getImmunity() && player.getName() != "")
        {
            leastPower = player.getPower();
            powerId = j;
            //cout << endl << "->"<< leastPower << " : " << powerId << endl << endl;
        }
    }

    // temp player
    player = *(team.getPlayers() + playerTurn);

    //number of votes per player
    while(player.getVotes())
    {
        // temp voted player
        Player votedPlayer = *(team.getPlayers() + powerId);
        // check
        if(votedPlayer.getName() != player.getName() && !votedPlayer.getImmunity() && votedPlayer.getName() != "")
        {
            Vote tempVote;
            tempVote.setVoted(votedPlayer.getName());
            tempVote.setReason(reasonWhy);
            votes.push_back(tempVote);
            player.setVotes(player.getVotes() - 1);
            //cout << player.getName() << " voted " << votedPlayer.getName() << endl;
        }
        // the player with the least power votes randomly
        else
        {
            // temp voted player
            Player votedPlayer = *(team.getPlayers() + rand()%10);
            // check
            if(votedPlayer.getName() != player.getName() && !votedPlayer.getImmunity() && votedPlayer.getName() != "")
            {
                Vote tempVote;
                tempVote.setVoted(votedPlayer.getName());
                tempVote.setReason("Random");
                votes.push_back(tempVote);
                player.setVotes(player.getVotes() - 1);
                //cout << player.getName() << " voted " << votedPlayer.getName() << endl;
            }
        }
    }
}

void Voting::votingProcess(Team &team)
{
    // check how many players have the team
    if (team.getNumberOfPlayers() == 2 )
    {
        cout << endl << "Only 2 players remaining.." << endl << endl;
        for ( int i = 0 ; i < 10 ; i ++)
        {
            Player player = *(team.getPlayers() + i);
            if (player.getImmunity() == false && player.getName() != "")
            {
                team.removePlayer(player.getName());
                cout << player.getName() << " has been ELIMINATED!" << endl << endl;
            }
        }
        return ;
    }

    if (team.getNumberOfPlayers() == 1 )
    {
        for ( int i = 0 ; i < 10 ; i ++)
        {
            Player player = *(team.getPlayers() + i);
            if (player.getImmunity() == true && player.getName() != "")
            {
                cout << "Only " << player.getName() << " left in this team! " << endl << endl;
            }
        }
        return ;
    }

    srand(time(NULL));

//    for (int i=0; i< 10 ; i++){
//        if (team.getPlayers()[i].getName() != ""){
//            cout << "***************"<< endl;
//            team.getPlayers()[i].status();
//            cout << "***************"<< endl;
//        }
//    }

    // players cast their votes
    for (int k = 0 ; k < 10 ; k++ )
    {
        switch (rand() % 3)
        {
        case 0:
            // players cast their votes randomly
            randomVoting(team,k);
            break;
        case 1:
            // voting based on least power
            lessPowerVoting(team,k);
            break;
        case 2:
            // voting based on gender randomly
            genderVoting(team, k);
            break;
        default:
            cout << "*Error*" << endl;
        }
    }

    // print votes vector
    for(unsigned int i = 0; i < votes.size(); i++)
    {
        votes[i].status();
        cout << endl;
    }

    // insert results into map
    for(unsigned int i = 0; i < votes.size(); i++)
    {
        if(results.find(votes[i].getVoted()) == results.end())
        {
            results[votes[i].getVoted()] = 0;
            //cout << "not found" << endl;
        }
        results[votes[i].getVoted()]++;
    }

    cout << endl << endl << " The results are: " << endl;

    // print results
    for(int i = 0; i < 10 ; i++)
    {
        //temp player
        Player player = *(team.getPlayers() + i);
        if(results.find(player.getName()) != results.end())
            cout << player.getName() << "  : " << results[player.getName()] << " vote(s)" << endl << endl;

    }
    //TEMP PLAYER
    Player *player;
    int max_votes = 0;
    int max_counter = 0;
    for(int i = 0 ; i < 10 ; i++)
    {
        //temp player
        player = (team.getPlayers() + i);
        // if this player exists in the map, meaning he has been voted
        if(results.find(player->getName()) != results.end())
        {
            //cout << "player found..." << endl;
            if(results[player->getName()] > max_votes)
            {
                //cout << "max votes found..." << endl;
                max_votes = results[player->getName()];
            }
        }
    }
    for(int i = 0; i < 10 ; i++)
    {
        //temp player
        player = (team.getPlayers() + i);
        // check if 2 or more players have the max votes
        if(results[player->getName()] == max_votes)
        {
            // remove from map, and set as candidate
            //cout << "setting candidate..." << endl;
            player->setCandidate(1);
            results.erase(player->getName());
            max_counter++;
        }
    }
    // if only one player has the max votes, select another one
    //cout << "second round..." << endl;
    if(max_counter == 1)
    {
        max_votes = 0;
        for(int i = 0 ; i < 10 ; i++)
        {
            //temp player
            player = (team.getPlayers() + i);
            if(results.find(player->getName()) != results.end())
            {
                //cout << "player found..." << endl;
                if(results[player->getName()] > max_votes)
                {
                    //cout << "max votes found..." << endl;
                    max_votes = results[player->getName()];
                }
            }
        }
        for(int i = 0; i < 10 ; i++)
        {
            //temp player
            player = (team.getPlayers() + i);
            // set as candidate
            if(results[player->getName()] == max_votes)
            {
                //cout << "setting candidate..." << endl;
                player->setCandidate(1);
                max_counter++;
            }
        }
    }
    // print candidates
    cout << " The candidates for elimination are:" << endl;

    for(int i = 0; i < 10; i++)
    {
        //temp player
        Player player = *(team.getPlayers() + i);
        if(player.getCandidate())
            cout << "-> " << player.getName() << "!" << endl << endl;
    }


    //select player to be eliminated
    //cout << max_counter << endl;
    // candidateNum counts down to 0, whoever gets 0 is eliminated
    int candidateNum = rand()%(max_counter);
    //cout << candidateNum << endl;
    for(int i = 0; i < 10; i ++)
    {
        player = (team.getPlayers() + i);
        if(player->getCandidate())
        {
            if(candidateNum == 0)
            {
                cout <<" At the end, " << player->getName() << " has been ELIMINATED!" << endl << endl;
                team.removePlayer(player->getName());
            }
            candidateNum--;
        }
    }

    // return Candidate and Votes to original values
    for(int i = 0; i < 10; i++)
    {
        player = (team.getPlayers() + i);
        if(player->getName() != "")
        {
            player->setCandidate(0);
            player->setVotes(1);
        }
    }

    // clear vector & map
    votes.clear();
    results.clear();

}
