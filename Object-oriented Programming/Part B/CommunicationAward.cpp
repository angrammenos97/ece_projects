#include "CommunicationAward.h"

using namespace std;

void CommunicationAward::status(){

    Award::status();

    cout << "Letters:" << endl;

    for (int i = 0;i < 5;i++){
        cout << letters[i] << endl;
    }

    cout << endl<< endl;

}
