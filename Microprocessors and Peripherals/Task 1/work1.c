__asm void isPalindromic(const char *src, unsigned int *result) {
	PUSH {r4, r5, r6} 	// Push values from r4, r5, r6 to stack since we are going to use them
	
	MOV r2, #0			// Initialize counter
counter_loop
	LDRB  r3, [r0, r2] 	// Load one byte (character) to r3
	CMP r3, #0			// Compare the character loaded with NULL character
	ADDNE r2, #1		// If character is not NULL, then increment counter by 1
	BNE counter_loop	// Do the same loop
	SUB r2, #1			// r2 now contains the size of string minus one, r2 will be used to to access characters from the end
	
	MOV r5, #0			// r5 is the counter used to access characters from the beginning
	MOV r6, #1			// r6 is the result (1 or 0) which defaults to 1
scan_loop
	CMP r5, r2			// Compare the two counters
	BPL end				// If r5>=r2, then further character comparisons are unnecessary

	LDRB r3, [r0, r5]	// Load character from where r5 is pointing
	LDRB r4, [r0, r2]	// Load character from where r2 is pointing
	
	CMP r3, r4			// Compare the two characters
	
	MOVNE r6, #0		// If they are not equal, then the result is 0
	BNE end				// If they are not equal, then break the loop
	
	ADD r5, #1			// If the loop is not broken, proceed by incrementing the r5-counter
	SUB r2, #1			// If the loop is not broken, proceed by decrementing the r2-counter
	B scan_loop			// Run the loop again until r5>=r2 or the loop breaks by not equal characters
	
end
	STR r6, [r1]		// Store r6 (result) to result pointer
	POP {r4, r5, r6}	// Pop values from stack to r4, r5, r6 that were pushed at the beginning of the function
	BX lr				// Return to main
}

int main() {
	unsigned int a = 0;
	const char my_str[] = "step on no pets";
	isPalindromic(my_str, &a);
	return 0;
}
