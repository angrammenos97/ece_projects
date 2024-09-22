#ifndef ROUND_H_INCLUDED
#define ROUND_H_INCLUDED

#include <iostream>
#include <string>

using namespace std;

class Round{

    int id;
    int duration;
    string winner;

public:

    Round(){id = 0; duration = 0; winner = "";}
    Round(int id, int duration, string winner){this->id = id;this->duration = duration; this->winner = winner;}

    void setId(int id){this->id = id;}
    void setDuration(int d){duration = d;}
    void setWinner(string w){winner = w;}

    int getId(){return id;};
    int getDuration(){return duration;}
    string getWinner(){return winner;}

    void status(){cout << "Round: " << id << " Duration: " << duration << " Winner: " << winner << endl;}
};


#endif // ROUND_H_INCLUDED
