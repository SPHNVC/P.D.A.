#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define SIZE 10

int main(int argc, char **argv)
{
	int arr[SIZE] = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
	int numprocs, procid, partner;
	int i = 0, myresult, ele = 1;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &procid);
	MPI_Status status;
	
	while (i < SIZE && ele != arr[i])
		i++;
		
	if(i < SIZE)
		myresult = i + 1;
	else
		printf("Not found");
	
	if(procid < numprocs / 2)
	{
		partner = numprocs/2 + procid;
		MPI_Send(arr, SIZE, MPI_INT, partner, 1, MPI_COMM_WORLD);
		MPI_Recv(&myresult, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
	}
	else if(procid >= numprocs / 2)
	{
		partner = procid - numprocs / 2;
		MPI_Recv(arr, SIZE, MPI_INT, partner, 1, MPI_COMM_WORLD, &status);
		MPI_Send(&myresult, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
	}
	
	
	printf("Position is: %d", myresult);

} 
