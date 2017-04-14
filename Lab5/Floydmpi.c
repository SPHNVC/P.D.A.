#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define SIZE 4

int main(int argc, char *argv[])
{
	int i,j,k;
	int a[SIZE][SIZE] = {
		{0, 2, 99, 10},
		{2, 0, 3, 99},
		{99, 3, 0, 1},
		{10, 99, 1, 0},
	};
	
	int costMatrix[SIZE][SIZE];    
	int numprocs, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	for (k = 0; k < SIZE; k++)
	{
		for (i = rank; i < SIZE; i = i + numprocs)
		{
			for (j = 0; j < SIZE; j++)
			{
				if (a[i][j] > a[i][k] + a[k][j])
					
					a[i][j] = a[i][k] + a[k][j];
			}
		}
		
		MPI_Allreduce(a, costMatrix, SIZE*SIZE, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		
		for (i = 0; i < SIZE; i++)
			for (j = 0; j < SIZE; j++)
				a[i][j] = costMatrix[i][j];		
		
		
	}

	MPI_Finalize();

	if (rank == 0)
	{
		for (i = 0; i < SIZE; i++)
		{
			for (j = 0; j < SIZE; j++)
			{
				printf("%d ", costMatrix[i][j]);
			}
			printf("\n");
		}
	}
		
}
