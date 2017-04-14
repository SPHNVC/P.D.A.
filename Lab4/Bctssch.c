#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAXSIZE 50

int main(int argc, char **argv)
{
	int myid, numprocs;	
	int arr[MAXSIZE], i, myresult = 0, result, ele = 23, x, low, high;
	FILE *fp;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if(0 == myid)
	{
		if(NULL == (fp = fopen("file.txt", "r")))
		{
			printf("Can`t open the file");
			exit(1);
		}
		for(i = 0; i < MAXSIZE; i++)
		{
			fscanf(fp, "%d", &arr[i]);
		}
			
	}

	MPI_Bcast(arr, MAXSIZE, MPI_INT, 0, MPI_COMM_WORLD);

	x = MAXSIZE / numprocs;
	low = myid * x;
	high = myid + x;
	for(i = low; i < high; i++)
	{
		if(ele == arr[i])
		{
			myresult = i + 1;
		}
	}
	
	MPI_Reduce(&myresult, &result, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	if(0 == myid)
	{
		printf("Maximum position is index: %d\n", result);
	}

	MPI_Finalize();
}

