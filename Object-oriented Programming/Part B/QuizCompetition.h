#ifndef QUIZCOMPETITION_H_INCLUDED
#define QUIZCOMPETITION_H_INCLUDED

#include "Competition.h"
#include "CommunicationAward.h"

using namespace std;

class QuizCompetition:public Competition
{
    CommunicationAward communicationAward;
    Round rounds[19];
    //int answers[10]= {4000,2,100000,23,12,1,200000,2,10,6500};
    static int answers[10];
    static string questions[10];
public:
    QuizCompetition() {

    };
    QuizCompetition(int id, string name, CommunicationAward communicationAward):Competition(id,name)
    {
        this -> communicationAward = communicationAward;
        this -> id = id;
        this -> name = name;
    };
    ~QuizCompetition()
    {
        cout << "Quiz Competition " << id << " destroyed!" << endl;
    }
    //getters
    string getQuestion(int i)
    {
        return questions[i];
    }
    int getAnswer(int i)
    {
        return answers[i];
    }
    //setters
    void setQuestion(int i, string a)
    {
        questions[i] = a;
    }
    void setAnswer(int i, int a)
    {
        answers[i] = a;
    }
    //status
    void status()
    {
        cout << "This is a Quiz Competition." << endl;
        Competition::status();
    }

    void compete(Team &team1,Team &team2)
    {
        for(int i = 0; i < 19; i++)
        {
            rounds[i].setDuration(20);
            rounds[i].setId(i+1);
        }
        int wins1 = 0, wins2 = 0;
        int qcounter, ans1, ans2;
        bool retry=0;               //case if both has the same answer
        Player player1, player2;
        srand(time(NULL));
        while(wins1 < 10 && wins2 < 10)
        {
            player1 = *(team1.getPlayers() + (rand() % team1.getNumberOfPlayers()));
            player2 = *(team2.getPlayers() + (rand() % team2.getNumberOfPlayers()));
            do
            {
                qcounter = rand() % 10;
                ans1 = rand() % 2*answers[qcounter];        //choose random num from 0-2*(answer of question)
                ans2 = rand() % 2*answers[qcounter];
                if(abs(ans1 - answers[qcounter]) < abs(ans2 - answers[qcounter]))
                {
                    rounds[wins1 + wins2].setWinner(player1.getName());
                    wins1++;
                    retry = 0;
                }
                else if(abs(ans1 - answers[qcounter]) > abs(ans2 - answers[qcounter]))
                {
                    rounds[wins1 + wins2].setWinner(player2.getName());
                    wins2++;
                    retry = 0;
                }
                else
                {
                    retry = 1;
                }
            }
            while(retry);
        }

        for(int i = 0; i < wins1 + wins2; i++)
        {

            rounds[i].status();
        }
        if(wins1 == 10)
        {
            cout << endl << "Final Score:" << endl << "Diashmoi: " << wins1 << endl << "Maxhtes: " << wins2 << endl;
            cout << "Team Diashmoi have won!" << endl;
            team1.setWins(team1.getWins() + 1);
        }
        else
        {
            cout << endl << "Final Score:" << endl << "Diashmoi: " << wins1 << endl << "Maxhtes: " << wins2 << endl;
            cout << "Team Maxhtes have won!" << endl;
            team2.setWins(team2.getWins() + 1);
        }
    }
};

string QuizCompetition::questions[10]= {"How many women has Spaliaras slept with?","How many Nobel prizes have been won by Greeks?","How many Tsangks are there in China?","Which is the number in the jersey of LeBron James?","How many stars are in Panos jersey?","How many books are in Choutos' Library? ","How many fans are out there loving Danos?","How many pounds are in one kilogram?","How much is the fish?","How many characters in the mandarene alphabet?"};
int QuizCompetition::answers[10]= {4000,2,100000,23,12,1,200000,2,10,6500};
#endif // QUIZCOMPETITION_H_INCLUDED
