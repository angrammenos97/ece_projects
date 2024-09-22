#ifndef VOTING_H_INCLUDED
#define VOTING_H_INCLUDED
#include <vector>
#include <map>
#include "Vote.h"
#include "Team.h"
// TO FILL UP

class Voting{


public:
    static vector<Vote>votes;
    static map<string, int>results;
    static void votingProcess(Team &team);
    static void genderVoting(Team &team , int k);
    static void randomVoting(Team &team , int k);
    static void lessPowerVoting(Team &team , int k);
};

#endif // VOTING_H_INCLUDED
