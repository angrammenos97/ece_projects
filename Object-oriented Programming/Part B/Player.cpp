#include "Player.h"

using namespace std;

Player::Player(){

    name = "";
    gender = female;
    age = 0;
    power = 100;
    hunger = 0;
    candidate = false;

}


Player::Player(string n, genderOption g, int a){

    name = n;
    gender = g;
    age = a;
    power = 100;
    hunger = 0;
    candidate = false;

}

Player::~Player(){

    cout << "Player " << name << " is destroyed." << endl;

}

string Player::getName(){
    return name;
}

int Player::getAge(){
    return age;
}

genderOption Player::getGender(){
    return gender;
}

float Player::getPower(){
    return power;
}

float Player::getHunger(){
    return hunger;
}

bool Player::getCandidate(){
    return candidate;
}

void Player::setName(string val){

    name = val;

}

void Player::setPower(float val){

    power = val;

}

void Player::setHunger(float val){

    hunger = val;

}

void Player::setCandidate(bool val){

    candidate = val;

}

void Player::work(){

    float value = 30 + (((float) rand()) / (float) RAND_MAX)*30;
    power *= (1 - (value/100));
    hunger += value;
    if (hunger > 100)
        hunger = 100;

    if (hunger < 0)
        hunger = 0;

}

void Player::eat(){

    float value = 10 + (((float) rand()) / (float) RAND_MAX)*30;
    power += value;
    hunger -= 80;

    if (power > 100)
        power = 100;

    if (hunger < 0)
        hunger = 0;

}

void Player::sleep(){
    power = 100;
}

void Player::compete(){

    float value = 20 + (((float) rand()) / (float) RAND_MAX)*30;
    power *= (1 - (value/100));
    hunger += 25;

    if (power < 0)
        power = 0;

    if (hunger > 100)
        hunger = 100;
}

void Player::status(){

    cout << "Name: " << name;
    cout << " Gender: ";
    if (gender == male)
        cout << "Male";
    else
        cout << "Female";
    cout << " Age: " << age;
    cout << " Power: " << power;
    cout << " Hunger: " << hunger;
    cout << " Candidate: ";
    if (candidate == 0)
        cout << "false";
    else
        cout << "true";
    cout << endl << endl;
}
