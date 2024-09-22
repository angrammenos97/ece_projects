#include "game.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ncurses.h>
#include "E9212.h"
char word[21];

void control_menu();

void random_word()
{
    int i=0, n, m, tmp;
    char c;
    FILE *f_in;
    f_in=fopen("words.txt","r");
    srand(time(NULL));
    m=rand();
    n=m*rand() % 854;
    printf("rand=%d\n",n);
    i=0;
    do{
        if(c=='\n') {
            i++;
        }
        if(i==n){
            tmp = fscanf(f_in,"%s",word);
            break;
        }
        else {
            tmp =fscanf(f_in,"%c",&c);
        }
    } while(i!=n);
    (void)tmp;
    printf("%s\n",word);
    fclose(f_in);
    game_body();
}

void input_word()
{
    clr_screen();
    printf("Player1 input a word up to 20 characters:");
    int tmp =scanf("%s",word);
    if(strlen(word)>20){
        printf("It's more than 20 charactrers!Try again\n");
        input_word();
    }
    game_body();
    (void)tmp;
}

int char_check(char x)
{
    char no_char[2][26]= {{'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'},
        {'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'}
    };
    for (unsigned int j=0; j<26; j++){
        if (no_char[0][j]==x || no_char[1][j]==x ){
            return 1;
        }
    }
    return 0;
}

void game_body()
{
    char x,tries[26]={0};       //x for character input and tries array to save the already written characters
    unsigned int n=0,counter=0,y=0,z=0,win=0,word_length=strlen(word); //counter for countering the tries, n for wrong characters counter , y to check if player2 found a character (1=TRUE 0=FALSE),z to check if player rewrite a character
    char *input=(char*)malloc(word_length * sizeof(char));
    for (unsigned int i=0;i<word_length;i++){
        input[i]='-';       // input array fill with *
    }
    input[word_length]=0;
    while (n<6){        //main game
        win=1;          //reseting value statements
        y=0;
        clr_screen();
        if(z==1){printf("You have already write this character!\n");}
        z=0;
        printf("Current status of word is:%s with %d life(s)\n",input,6-n);
        printf("You have already write: %s\n",tries);
        hangman(n);
        printf("Player2 write a character:");
        x=getch();
        if(x==27){control_menu();}      //to cancel the current game
        while(!char_check(x)){       //if player2 write non character
            printf("\nPlease try again.\n");
            printf("Player2 write a character:");
            x=getch();
            if(x==27){control_menu();}
        }
        for(unsigned int j=0; j<counter; j++){
            if(x==tries[j]){        //found character that already is written
                    z=1;
                    y=1;
            }
        }
        if(z==0){tries[counter]=x;} else {counter--;}       //insert new character as written
        for (unsigned int i=0;i<word_length;i++){       //change * to found character
            if (word[i]==x){
                    input[i]=word[i];
                    y=1;
            }
            if(input[i]=='-'){win=0;}       //if there is still character to be found return win FALSE
        }
        if(win==1){
            clr_screen();
            printf("Player2 win! The word is %s.\n",word);      //winning output
            win_game();
            break;
        }
        counter++;
        if(y==0){n++;}      //if character didnt found decrease one life
    }
    if(win==0){
            clr_screen();
            printf("Player2 lost. The word is %s.\n",word); //output of loser
            hangman(6);
    }
    free(input);
}

int arrows()
{
    switch(getch()) {
    case 72: return 1;   // key up
        break;
    case 80: return 2;  // key down
        break;
    case 77: return 3;  // key right
        break;
    case 75: return 4;   // key left
        break;
    default :arrows();}
    return 0;
}

void control_menu()
{
    int state,ext;
    state=0;
    ext=0;
    main_menu(3,1);
    while ((getch())!='\r'){state=main_menu(arrows(),0);}
    if (state==0){random_word();printf("Press any key to continue..");getch();control_menu();}
    else if (state==1){input_word();printf("Press any key to continue..");getch();control_menu();}
    else if (state==2){printf("Setting coming soon!\n");}
    else if (state==3){exit_menu(0,1);ext=0;while ((getch())!='\r'){ext=exit_menu(arrows(),0);};if(ext==0){exit(0);}else{control_menu();}}
    else {printf("Error\n");}
}

int main()
{
    initscr();            // Initialize ncurses mode
    noecho();             // Disable echoing
    cbreak();             // Disable line buffering (so no need for Enter)
    control_menu();
	return 0;
}
