#ifndef VOTE_H_INCLUDED
#define VOTE_H_INCLUDED
#include <cstring>
#include <iostream>

using namespace std;
// TO FILL UP

class Vote{
 string voted;
 string reason;
public:
    Vote(){voted = " "; reason = " ";};
    Vote(string voted, string reason){this -> voted = voted; this -> reason = reason;}
    ~Vote (){/*cout << "A vote has been destroyed"<<endl;*/}
    string getVoted(){return voted;}
    string getReason(){return reason;}
    void setVoted(string voted){this -> voted = voted;}
    void setReason(string reason){this -> reason = reason;}
    void status(){
        cout << "A player voted: " << voted << endl;
        cout << "Reason: " << reason << endl;
    }
};

#endif // VOTE_H_INCLUDED
