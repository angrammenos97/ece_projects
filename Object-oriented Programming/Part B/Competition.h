#ifndef COMPETITION_H_INCLUDED
#define COMPETITION_H_INCLUDED
#include <cstring>

using namespace std;

class Competition
{
protected:
    int id;
    string name;
    string winner;

public:
    Competition()
    {
        id = 0;
        name = " ";
        winner = " ";
    }
    Competition(int a, string b)
    {
        id = a;
        name = b;
    }
    ~Competition()
    {
        cout << "Competition " << id << " destroyed!" << endl;
    }
    // getters!
    int getId()
    {
        return id;
    }
    string getName()
    {
        return name;
    }
    string getWinner()
    {
        return winner;
    }
    // setters!
    void setId(int id)
    {
        this ->id=id;
    }
    void setName(string name)
    {
        this -> name=name;
    }
    void setWinner(string winner)
    {
        this -> winner=winner;
    }
    // status!
    void status()
    {
        cout << "Competition Id: " << id << endl << "Competition name: " << name << endl << "Winner " << winner << endl;
    }

};

#endif // COMPETITION_H_INCLUDED
