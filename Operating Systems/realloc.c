#include <stdio.h>
#include <stdlib.h>

/* ============ (i) VECTOR: REALLOC AT EACH UPDATE */
struct Stack {
	int top;
	int* array;
};

struct Stack* createStack()
{
	struct Stack* new_stack = (struct Stack*)malloc(sizeof(struct Stack));
	new_stack->array = NULL;
	new_stack->top = 0;
	return new_stack;
}

int isEmpty(struct Stack* stack)
{
	if (stack->top <= 0)
		return 1;
	return 0;
}

void push(struct Stack* stack, int item)
{
	stack->top++;
	stack->array = (int*)realloc(stack->array, stack->top * sizeof(int));
	*(stack->array + stack->top - 1) = item;
}

int pop(struct Stack* stack)
{
	if (isEmpty(stack))
		exit(1);
	int pop_value = *(stack->array + stack->top - 1);
	stack->top--;
	stack->array = (int*)realloc(stack->array, stack->top * sizeof(int));
	return pop_value;
}

/* ============ (ii) VECTOR: REALLOC AT POWERS OF 2 */
struct StackFast {
	int top;
	int* array;
};

struct StackFast* createStackFast()
{
	struct StackFast* new_stack = (struct StackFast*)malloc(sizeof(struct StackFast));
	new_stack->array = NULL;
	new_stack->top = 0;
	return new_stack;
}

int isEmptyFast(struct StackFast* stack)
{
	if (stack->top <= 0)
		return 1;
	return 0;
}

void pushFast(struct StackFast* stack, int item)
{
	stack->top++;
	int needed_size = 1;
	while (needed_size < stack->top)
		needed_size *= 2;
	stack->array = (int*)realloc(stack->array, needed_size * sizeof(int));
	*(stack->array + stack->top - 1) = item;
}

int popFast(struct StackFast* stack)
{
	if (isEmptyFast(stack))
		exit(1);
	int pop_value = *(stack->array + stack->top - 1);
	stack->top--;
	int needed_size = 1;
	while (needed_size < stack->top)
		needed_size *= 2;
	stack->array = (int*)realloc(stack->array, needed_size * sizeof(int));
	return pop_value;
}
//////////////////////////////////////////
int main()
{
	/*int num_loop = 10;
	struct Stack* simple_stack = createStack();
	for (int i = 1; i <= num_loop; i++) {
		push(simple_stack, i);
		//printf("The last element of the simple stack is %i.\n", pop(simple_stack));
	}
	//simple_stack->array = (int*)simple_stack->top;
	for (int i = 1; i <= num_loop; i++)
		printf("The last element of the simple stack is %i.\n", pop(simple_stack));*/

	int num_loop = 100;
	struct StackFast* simple_stack = createStackFast();
	for (int i = 1; i <= num_loop; i++) {
		pushFast(simple_stack, i);
		//printf("The last element of the simple stack is %i.\n", pop(simple_stack));
	}
	//simple_stack->array = (int*)simple_stack->top;
	for (int i = 1; i <= num_loop; i++)
		printf("The last element of the simple stack is %i.\n", popFast(simple_stack));


	printf("DONE!\n");
	return 0;
}
