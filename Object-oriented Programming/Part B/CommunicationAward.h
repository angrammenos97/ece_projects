/* амастасиос цяаллемос:
 *     аел: 9212
 *     email: avgramme@ece.auth.gr
 * леяйоуяиос силос:
 *     AEM: 9133
 *     email: merkours@ece.auth.gr
 */


#ifndef COMMUNICATIONAWARD_H_INCLUDED
#define COMMUNICATIONAWARD_H_INCLUDED

#include "Award.h"

using namespace std;

class CommunicationAward : public Award{

    string letters[5] = {"Dose pono, se stirizo", "Se agapo", "Mou leipeis", "Panara zoume winner na se doume", "Eisai mia savrrrrrrrrra pou kyliesai kato"};

public:

    CommunicationAward(){;}
    CommunicationAward(string n, bool b) : Award(n,b){};

    void createLetters(int val);
    string *getLetters(){return letters;}

    void status();
};

#endif // COMMUNICATIONAWARD_H_INCLUDED
