#include <stdio.h>
#include <stdlib.h>
int state=1;
int ext=0;

void hangman(int n)
{
	printf ("\n++==========\n");
	printf ("|| //      |\n");
	if (n>0){printf ("||//      _|_\n");} else {printf ("||//       |\n");}
	if (n>3){printf ("||/ \\\\   |. .|    //\n");} else if (n>2){printf ("||/ \\\\   |. .|\n");} else if (n>0){printf ("||/      |. .|\n");} else {printf ("||/\n");}
	if (n>3){printf ("||   \\\\  | O |   //\n");} else if (n>2){printf ("||   \\\\  | O |\n");} else if (n>0){printf ("||       | O |     \n");} else {printf ("||\n");}
	if (n>3){printf ("||    \\\\  | |   //\n");} else if (n>2){printf ("||    \\\\  | |\n");} else if (n>0){printf ("||        | |   \n");} else {printf ("||\n");}
	if (n>3){printf ("||     \\\\--  --//\n");} else if (n>2){printf ("||     \\\\--  ----\n");} else if (n>1){printf ("||     ----  ----\n");}else {printf ("||\n");}
	if (n>1){printf ("||      |      |\n");} else {printf ("||\n");}
	if (n>1){printf ("||      |      |\n");} else {printf ("||\n");}
	if (n>1){printf ("||      |      |\n");} else {printf ("||\n");}
	if (n>1){printf ("||      |      |\n");} else {printf ("||\n");}
	if (n>4){printf ("||       ==o===\n");} else if (n>1){printf ("||       ------\n");} else {printf ("||\n");}
	if (n>5){printf ("||      /      \\\n");} else if (n>4){printf ("||      /   /\n");} else {printf ("||\n");}
	if (n>5){printf ("||      |  /\\  |\n");} else if (n>4){printf ("||      |  /\n");} else {printf ("||\n");}
	if (n>5){printf ("||      |  ||  |\n");} else if (n>4){printf ("||      |  |\n");} else {printf ("||\n");}
	if (n>5){printf ("||      |  ||  |\n");} else if (n>4){printf ("||      |  |\n");} else {printf ("||\n");}
	if (n>5){printf ("||      |  ||  |\n");} else if (n>4){printf ("||      |  |\n");} else {printf ("||\n");}
	if (n>5){printf ("||     (___||___)\n");} else if (n>4){printf ("||     (___|\n");} else {printf ("||\n");}
	printf ("||\n");
	printf ("||\n");
	printf ("||\n");
	printf ("||__________________\n");
	printf ("|___________________|\n\n");
}

void win_game()
{
    printf("      ___\n");
    printf("\\\\   |^ ^|    //\n");
    printf(" \\\\  | O |   //\n");
    printf("  \\\\  | |   //\n");
    printf("   \\\\--  --//\n");
    printf("    |      |\n");
    printf("    |      |\n");
    printf("    |      |\n");
    printf("    |      |\n");
    printf("     ==o===\n");
    printf("    /      \\\n");
    printf("    |  /\\  |\n");
    printf("    |  ||  |\n");
    printf("    |  ||  |\n");
    printf("    |  ||  |\n");
    printf("   (___||___)\n\n");
}

void clr_screen()
{
    int tmp = system("@cls||clear");
    printf (" #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#\n");
    printf (" #											       #\n");
    printf (" #  ##     ##        ####  ###     ##     #######    #####      ####        ####  ###     ##   #\n");
    printf (" #  ##     ##       ## ##  ####    ##   ##       ##  ##  ##   ##  ##       ## ##  ####    ##   #\n");
    printf (" #  ##     ##      ##  ##  ## ##   ##  ##            ##   ## ##   ##      ##  ##  ## ##   ##   #\n");
    printf (" #  #########     ##   ##  ##  ##  ##  ##     #####  ##    ###    ##     ##   ##  ##  ##  ##   #\n");
    printf (" #  ##     ##    ## #####  ##   ## ##  ##     #  ##  ##     #     ##    ## #####  ##   ## ##   #\n");
    printf (" #  ##     ##   ##     ##  ##    ####  ##        ##  ##           ##   ##     ##  ##    ####   #\n");
    printf (" #  ##     ##  ##      ##  ##     ###    ########    ##           ##  ##      ##  ##     ###   #\n");
    printf (" #											       #\n");
    printf (" #*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#\n\n");
    (void)tmp;
}

int main_menu(int arrow,int bgn)
{
    if (bgn==1){state=0;}
    if (arrow==2 && state<3){state++;}else if (arrow==2 && state==3){state=0;}
    if (arrow==1 && state>0){state--;}else if (arrow==1 && state==0){state=3;}
    clr_screen();
    if (state==0){printf("\t\t\t\t\t> 1 PLAYER <\n");} else {printf("\t\t\t\t\t1 PLAYER\n");}
    printf("\n\n");
    if (state==1){printf("\t\t\t\t\t> 2 PLAYERS <\n");} else {printf("\t\t\t\t\t2 PLAYERS\n");}
    printf("\n\n");
    if (state==2){printf("\t\t\t\t\t> SETTINGS <\n");} else {printf("\t\t\t\t\tSETTINGS\n");}
    printf("\n\n");
    if (state==3){printf("\t\t\t\t\t> EXIT <\n");} else {printf("\t\t\t\t\tEXIT\n");}
    return state;
}

int exit_menu(int arrow,int bgn)
{
    if (bgn==1){ext=0;}
    if (arrow==3 && ext==0){ext++;} else if (arrow==3 && ext==1){ext=0;}
    if (arrow==4 && ext==1){ext--;} else if (arrow==4 && ext==0){ext=1;}
    clr_screen();
    printf("\n\n\t\t\tAre you sure that you want to exit?\n");
    if(ext==0){printf("\t\t\t\t> YES <");} else {printf("\t\t\t\tYES");}
    if(ext==1){printf("\t\t> NO <\n");} else {printf("\t\tNO\n");}
    return ext;
}
