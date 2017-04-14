#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define SIZE 50

int main(int argc,char **argv)
{
	
	int rank, numprocs, source, sendcount, recvcount, i, ele = 23;
	int sendbuf[SIZE] = {23, 42, 54, 1, 4, 14, 24, 23, 92, 91, 18, 39, 38, 23}, recvbuf;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	if(numprocs == SIZE)
	{
		source = 1;
		sendcount = SIZE;
		recvcount = 0;
	}

	MPI_Scatter(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, source, MPI_COMM_WORLD);
	
	for(i = 0; )
	
	for(i = 0; i < SIZE; i++)
	{
		if(ele == recvbuf)
		{
			recvcount++;
		}
	}

	MPI_Gatter(sendbuf, sendcount, MPI_INT, recvbuf, recvcount, MPI_INT, source, MPI_COMM_WORLD);

	printf("Index is: %d", recvcount);

	MPI_Finalize();
}

