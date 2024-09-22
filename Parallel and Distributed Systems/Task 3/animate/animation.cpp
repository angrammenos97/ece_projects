#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <core.hpp>
#include <highgui.hpp>

#define mSPF  1000	// milliseconds per frame

using namespace cv;

char *input_file = NULL;
int npd = 0;	// Number of Points per Dimension
int nk = 0;		// Number of Iterations
int eximg = 0;	// Export images

void help(int argc, char *argv[]);
void import_data(char *G);

int main(int argc, char *argv[])
{
	help(argc, argv);

	if (input_file == NULL) {
		printf("Give input binary file!\n");
		exit(1);
	}
	printf("Key mapping:\n");
	printf("space\t: Play/Pause\n");
	printf("p\t: Previous frame\n");
	printf("esc|q\t: Exit\n");
	printf("Other\t: Goes to next frame\n");
	printf("Note: When press p the playback pauses.\n");

	char *all_frames = (char*)malloc(npd*npd*nk * sizeof(char));
	import_data(all_frames);

	// Export to images
	if (eximg)
		for (int i = 0; i < nk; i++) {
			char flname[100];
			sprintf(flname, "%s-%i.jpg", flname, i);
		}

	printf("n=%i k=%i\n", npd, nk);
	Mat frame(npd, npd, CV_8UC1);
	int millis_per_frame = mSPF;
	int curr_frame = 0;
	while (true) {
		memcpy(frame.ptr(0), (all_frames + curr_frame * npd*npd), npd*npd);
		if (eximg) {	// Export image
			char flname[100];
			sprintf(flname, "./%s-%i.jpg", input_file, curr_frame);
			imwrite(flname, frame);
		}
		printf("frame #%i\n", curr_frame);
		imshow("Frame", frame);
		int c = waitKey(millis_per_frame);
		if (c == 32) { // space button
			if (millis_per_frame == mSPF)	// pause
				millis_per_frame = -1;
			else
				millis_per_frame = mSPF;	// play

		}
		else if (c == 112) { // p button
			millis_per_frame = -1;			// pause
			if (curr_frame > 0)
				curr_frame -= 1;	// go back a frame
		}
		else if (c == 113 || c == 27) // q or esc button
			break;							// exit
		else if (curr_frame < nk - 1)	// go to next frame
			curr_frame++;
		else {
			millis_per_frame = -1;
			printf("End\n");
		}
	}

	printf("Exiting...\n");
	destroyAllWindows();
	frame.release();
	free(all_frames);
	return 0;
}

void help(int argc, char *argv[])
{
	if (argc > 1) {
		for (int i = 1; i < argc; i += 2) {
			if (*argv[i] == '-') {
				if (*(argv[i] + 1) == 'f')
					input_file = argv[i + 1];
				else if (*(argv[i] + 1) == 'n')
					npd = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'k')
					nk = atoi(argv[i + 1]);
				else if (*(argv[i] + 1) == 'i')
					eximg = atoi(argv[i + 1]);
				else {
					help(1, argv);
					return;
				}
			}
			else {
				help(1, argv);
				return;
			}
		}
		return;
	}
	printf("Flags to use:\n");
	printf("-f [File]\t:Input file of points\n");
	printf("-n [Number]\t:Number of points per dimension \n");
	printf("-k [Frames]\t:Number of frames \n");
	printf("-i [0|1]\t:To export each frame to jpg image \n");
}

void import_data(char *G)
{
	printf("Importing data from %s. ", input_file);
	FILE *f = fopen(input_file, "rb");
	int *bufi = (int*)malloc(sizeof(int));
	for (int i = 0; i < npd*npd*nk; i++) {
		fread(bufi, sizeof(int), 1, f);
		*(G + i) = (char)*bufi;
	}
	fclose(f);
	free(bufi);
}