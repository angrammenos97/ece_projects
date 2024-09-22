#include <iostream>
#include <string>
#include "Team.h"
#include "Player.h"
#include <time.h>
#include <stdlib.h>


Player::Player()
{
    name = " ";
    gender = " ";
    profession = " ";
    item = " ";
    age = 0;
    wins = 0;
    hunger = 0.0;
    strength = 100.0;
    team = -1;
}
// This function is not used
Player::Player(int a, int t, string n, string g, string p, string i)
{
    team = t;
    name = n;
    gender = g;
    profession = p;
    item = i;
    age = a;
    wins = 0;
    hunger = 0.0;
    strength = 100.0;
}
Player::~Player()
{
    cout << "Player " << name << " destroyed!" << endl;
}
void Player::setAge(int a)
{
    age = a;
}
void Player::setWins(int a)
{
    wins = a;
}
void Player::setName(string a)
{
    name = a;
}
void Player::setGender(string a)
{
    gender = a;
}
void Player::setProfession(string a)
{
    profession = a;
}
void Player::setItem(string a)
{
    item = a;
}
void Player::setHunger(float a)
{
    hunger = a;
}
void Player::setStrength(float a)
{
    strength = a;
}
void Player::setTeam(int a)
{
    team = a;
}

int Player::getAge()
{
    return age;
}
int Player::getWins()
{
    return wins;
}
string Player::getName()
{
    return name;
}
string Player::getGender()
{
    return gender;
}
string Player::getProfession()
{
    return profession;
}
string Player::getItem()
{
    return item;
}
float Player::getHunger()
{
    return hunger;
}
float Player::getStrength()
{
    return strength;
}

int Player::getTeam()
{
    return team;
}
void Player::eat()
{
    srand(time(NULL)); // init random number generator
    hunger = hunger - 80.0;
    if(hunger<0.0) // lower limit
        hunger = 0.0;
    strength = strength + (rand() % 30 + 10);
    if(strength > 100.0) // upper limit
        strength = 100.0;
}
void Player::work()
{
    float tempS = 0;
    float tempH = 0;
    srand(time(NULL)); // init random number generator
    strength = tempS; // check if he/she can work
    strength = strength - strength * (rand() % 30 + 30) / 100;
    if(strength < 0.0)
    {
        cout << "Player " << name << " does not have enough strength!" << endl;
        strength = tempS;
    }
    hunger = tempH; // check if he/she can work
    hunger = hunger + hunger * 0.2;
    if(hunger > 100.0)
    {
        cout << "Player " << name << " is too hungry!" << endl;
        hunger = tempH;
    }
}
void Player::sleep()
{
    strength = 100.0;
}
