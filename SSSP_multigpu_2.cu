#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <climits>
#include "graph.hpp"

#define cudaCheckError() {                                             \
 cudaError_t e=cudaGetLastError();                                     \
 if(e!=cudaSuccess) {                                                  \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
   exit(0);                                                            \
 }                                                                     \
}

// structure representing the replica node
struct replicanode
{
	int nodeid;	// node id
	int dist;	// shortest dist of the replica node from the source vertex
};

// we need to push this inits to our library.cuda file
template <typename T>
__global__ void initKernel0(T* init_array, T id, T init_value) { // MOSTLY 1 thread kernel
  init_array[id]=init_value;
}


template <typename T>
__global__ void initKernel(unsigned V, T* init_array, T init_value) {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array[id]=init_value;
  }
}

template <typename T1, typename T2>
__global__ void initKernel2( unsigned V, T1* init_array1, T1 init_value1, T2* init_array2, T2 init_value2)  {
  unsigned id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id < V) {
    init_array1[id]=init_value1;
    init_array2[id]=init_value2;
  }
}

__global__ void Compute_SSSP_kernel(int dnum, int * gpu_offset_array ,
  int * gpu_edge_list ,
  int* gpu_weight,
  int * gpu_dist,
  int V,
  bool * gpu_modified_prev,
  bool * gpu_modified_next,
  bool * gpu_finished, 
  replicanode *gpu_replica_info,
  int perdevicevertices )
{
  //unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);	// changes here
  unsigned int v = id;
  //if ( id >= (dnum*perdevicevertices ) && id < ( (dnum+1)*perdevicevertices )  )
  if( id >= (dnum*perdevicevertices ) && id < ( (dnum+1)*perdevicevertices ) )
  {
   if (gpu_modified_prev[id] ){

	printf( "%d " , id );
      for (int edge = gpu_offset_array[id]; edge < gpu_offset_array[id+1]; edge ++)
      {
        int nbr = gpu_edge_list[edge] ;
        int e = edge;
        int dist_new;
        if(gpu_dist[id] != INT_MAX)
          dist_new = gpu_dist[v] + gpu_weight[e];

      // If neighbour is the internal node and not the boundary node 
      	if( nbr >= (dnum*perdevicevertices ) && nbr < ( (dnum+1)*perdevicevertices )  ){
	 if (gpu_dist[nbr] > dist_new)
         {
           atomicMin(&gpu_dist[nbr] , dist_new);
           gpu_modified_next[nbr]=true;
           *gpu_finished = false ;
         }
	}
	else{		// its the boundary node so update the distance in the replica
	 if ( gpu_replica_info[nbr].dist > dist_new)
         {
           atomicMin(&gpu_replica_info[nbr].dist , dist_new);
           atomicMin(&gpu_dist[nbr] , dist_new);
	   gpu_modified_next[nbr]=true;
           *gpu_finished = false ;
         }
	}	
         
      }
   }
  }

}
  void SSSP(int* offset_array , int* edge_list , int* cpu_edge_weight  , int src ,int V, int E , bool printAns)
{

  // Somewhere on CPU
  int devicecount;
  cudaGetDeviceCount(&devicecount);
  std::cout << devicecount << "\n";

  //CSR VARS
  int * gpu_offset_array[devicecount];
  for(int i=0;i<devicecount;i++)
  {	
  cudaSetDevice(i);
  cudaMalloc(&gpu_offset_array[i],sizeof(int) *(1+V));
  }

  int * gpu_edge_list[devicecount];
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_edge_list[i],sizeof(int) *(E));
  }
	
  int * gpu_edge_weight[devicecount];
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_edge_weight[i],sizeof(int) *(E));
  }

  // RESUT VAR
  int * gpu_dist[devicecount];
  int dist[devicecount][V];
  
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_dist[i],sizeof(int) *(V));
  }

  // EXTRA VARS
  bool cpu_modified_prev[devicecount][V];
  bool * gpu_modified_prev[devicecount];
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_modified_prev[i],sizeof(bool) *(V));
  }

  bool * gpu_modified_next[devicecount];
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_modified_next[i],sizeof(bool) *(V));
  }

  bool * gpu_finished[devicecount];
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_finished[i],sizeof(bool) *(1));
  }
  printf("Allocation finished\n");

  unsigned int block_size=V;
  unsigned int num_blocks=1;

  // Launch Config is ready!
  if ( V > 512 ) {
    block_size = 512;
    num_blocks = (V+block_size-1) / block_size; // avoid ceil fun call
  }
  std::cout<< "nBlock:" << num_blocks  << '\n';
  std::cout<< "threadsPerBlock:" << block_size  << '\n';
  // This comes from attach propoety
  //~ with two init1
  //~ initKernel<int> <<<num_blocks,block_size>>>(V,gpu_dist, INT_MAX);
  //~ initKernel<bool><<<num_blocks,block_size>>>(V,gpu_modified_prev, false);
  //~ with single init2
  
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  initKernel2<int,bool> <<<num_blocks,block_size>>>(V,gpu_dist[i], INT_MAX,gpu_modified_prev[i], false);
  }

  // This comes from DSL. Single thread kernel
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  initKernel0<int> <<<1,1>>>(gpu_dist[i], src,0);
  }

  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  initKernel0<bool> <<<1,1>>>(gpu_modified_prev[i], src,true);
  }

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  // CSR
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMemcpy (gpu_offset_array[i]  , offset_array    , sizeof(int)   *(1+V), cudaMemcpyHostToDevice);
  }

  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMemcpy (gpu_edge_list[i]     , edge_list       , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);
  }  

  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMemcpy (gpu_edge_weight[i]   , cpu_edge_weight , sizeof(int)   *(E)  , cudaMemcpyHostToDevice);
  }

  // COMES FROM DSL so CPU VARIABLE
  bool* finished = new bool[1];
  *finished = false; // to kick start

  unsigned int extra_blocks=0;
  extra_blocks = (num_blocks % devicecount);

  
  // Creating the replica nodes here...
  replicanode replica_info[devicecount][V];		// storing the replica nodes on CPU
  replicanode *gpu_replica_info[devicecount];		// storing the replica nodes on GPU

  for(int i=0;i<devicecount;i++)
  {
	cudaSetDevice(i);
  	cudaMalloc(&gpu_replica_info[i],sizeof(replicanode) *(V));
  }

  //for(int i=0;i<V+1;i++)
  //	printf( "%d ", offset_array[i] );
  //printf("\n");

  //for(int i=0;i<E;i++)
  //	printf( "%d ", edge_list[i] );
  //printf("\n");

  int perdevicevertices;
  perdevicevertices = V / devicecount;
  
  for(int i=0;i<devicecount;i++)
  {
	// for all the vertices handle by the current device..
	for(int j = i*perdevicevertices ; j < (i+1)*perdevicevertices ; j++)
	{
		// check their neighbours
		for(int k=offset_array[j] ; k<offset_array[j+1] ; k++ )
		{
			// if this is true then edge_list[k] is a outside node. We need to replicate it.
			if( edge_list[k] > (i+1)*perdevicevertices || edge_list[k] < i*perdevicevertices )
			{
				replicanode temp;
				temp.nodeid = edge_list[k];	// id of the node to be replicated
				if( temp.nodeid != 0 )						
					temp.dist = INT_MAX;		// initial dist of the replicated node is INT_MAX;
				else
					temp.dist = 0;			// corner case
				replica_info[i][edge_list[k]] = temp;
			}
		}
	}
  }


  // Partitioning ends here..

  int k =0; /// We need it only for count iterations attached to FIXED pt
  while ( !(*finished) )
  {
    //~ finished[0]=true;   /// I guess  we do not need this line overwrritten in memcpy below
    for(int i=0;i<devicecount;i++)
    {
    cudaSetDevice(i);
    initKernel<bool> <<< 1, 1>>>(1, gpu_finished[i], true);
    }
	
    // copying the replica nodes to the GPU from CPU
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);	
	cudaMemcpy(gpu_replica_info[i], replica_info[i],  sizeof(replicanode)*V, cudaMemcpyHostToDevice);
    }
	
    // kernel call on each of the GPU -- Processing their chunck of vertices independently
    for(int i=0;i<devicecount;i++)
    {
	cudaSetDevice(i);	
	//cudaSetDevice(0);
	Compute_SSSP_kernel<<<num_blocks , block_size>>>(i, gpu_offset_array[i], gpu_edge_list[i],  gpu_edge_weight[i] ,gpu_dist[i] ,V , gpu_modified_prev[i], gpu_modified_next[i], gpu_finished[i], gpu_replica_info[i], perdevicevertices );
    }
    // explicity handling the last kernel call
    //cudaSetDevice(devicecount-1);
    //cudaSetDevice(0);
    //Compute_SSSP_kernel<<<num_blocks, block_size>>>(devicecount-1, gpu_offset_array[devicecount-1], gpu_edge_list[devicecount-1],  gpu_edge_weight[devicecount-1] ,gpu_dist[devicecount-1] ,V , gpu_modified_prev[devicecount-1], gpu_modified_next[devicecount-1], gpu_finished[devicecount-1], gpu_replica_info[devicecount-1], perdevicevertices );

    // copying the replica nodes to the GPU from CPU
    //for(int i=0;i<devicecount;i++)
    //{
    //    cudaSetDevice(i);
    //    cudaMemcpy(replica_info[i], gpu_replica_info[i],  sizeof(replicanode)*V, cudaMemcpyDeviceToHost);
    //}
	
	printf("Hellooo\n");
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
    	initKernel<bool><<<num_blocks,block_size>>>(V, gpu_modified_prev[i], false);
    }
        printf("Hellooo\n");
    
    bool cpuarr_finished[devicecount];
    bool* currvalue;
    currvalue = (bool *)malloc(sizeof(bool) *(1));

    printf("Hellooo\n");
    
    for(int i=0;i<devicecount;i++)
    {
	printf("%d ", i);
        cudaSetDevice(i);
    	cudaMemcpy(currvalue, gpu_finished[i],  sizeof(bool) *(1), cudaMemcpyDeviceToHost);		// Problem : finished should be a 1-D array
    	cpuarr_finished[i] = *currvalue;
    }

    printf("Hellooo\n");

    *finished = true;
	
    for(int i=0;i<devicecount;i++)
	if( cpuarr_finished[i] == false )
		*finished = false;
/*	
    for( int j=0;j<devicecount; j++ )
        for (int i = 0; i <V; i++) 
        {
                printf("%d %d\n", i, dist[j][i]);
        }
    printf("\n\n");

	for(int i=0;i<devicecount;i++)
		printf( "%d ", cpuarr_finished[i] );
	printf("\n\n");
*/
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
    	bool *tempModPtr  = gpu_modified_next[i];
    	gpu_modified_next[i] = gpu_modified_prev[i];
    	gpu_modified_prev[i] = tempModPtr;
    }

    printf("I am in loop\n");
    
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
        // copying the GPU computed dist gpu_dist to CPU for updation purpose
	cudaMemcpy(dist[i],gpu_dist[i] , sizeof(int) * (V), cudaMemcpyDeviceToHost);	
	cudaMemcpy(cpu_modified_prev[i],gpu_modified_prev[i] , sizeof(bool) * (V), cudaMemcpyDeviceToHost);			
    }
	
    
    // update the CPU replica_info with the latest dist compute from dist array
    for(int i=0;i<devicecount;i++)
    {
	for(int j=0;j<V;j++)
	{
		for(int d=0;d<devicecount;d++)
		{
			//if( replica_info[d][j].dist > dist[i][j] )

			if( dist[d][j] > dist[i][j] )
			{
				replica_info[d][j].dist = dist[i][j];
				dist[d][j] = dist[i][j];	
				//cpu_modified_prev[d][j] = cpu_modified_prev[i][j];		// instead of doing this
			}
	
		}
	}
    }


    for(int j=0;j<V;j++)
    {
	bool currval  = false;
	for(int i=0;i<devicecount;i++)
	{
		currval = currval || cpu_modified_prev[i][j];
	}

	for(int i=0;i<devicecount;i++)
	{
		cpu_modified_prev[i][j] = currval;
	}
    }
    
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
        // copying the CPU updated dist to GPU for updation purpose
	cudaMemcpy(gpu_dist[i], dist[i] , sizeof(int) * (V), cudaMemcpyHostToDevice);		
	cudaMemcpy(gpu_modified_prev[i], cpu_modified_prev[i] , sizeof(bool) * (V), cudaMemcpyHostToDevice);		
    }

    ++k;
    if(k==V)    // NEED NOT GENERATE. DEBUG Only
    {
      std::cout<< "THIS SHOULD NEVER HAPPEN" << '\n';
      exit(0);
    }
      printf("Hi\n");
  }

  //STOP TIMER
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms \nIterations:%d\n", milliseconds,k);

  cudaCheckError()

  // PRINT THE OUTPUT vars
  if(printAns) 
  {
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
    	cudaMemcpy(dist[i],gpu_dist[i] , sizeof(int) * (V), cudaMemcpyDeviceToHost);
    }

    for( int j=0;j<devicecount; j++ )
    	for (int i = 0; i <V; i++) 
	{
      		printf("%d %d\n", i, dist[j][i]);
    	}
    printf("\n\n");
  }

  //~ char *outputfilename = "output_generated.txt";
  //~ FILE *outputfilepointer;
  //~ outputfilepointer = fopen(outputfilename, "w");
  //~ for (int i = 0; i <V; i++)
  //~ {
    //~ fprintf(outputfilepointer, "%d  %d\n", i, dist[i]);
  //~ }

}


// driver program to test above function
int main(int argc , char ** argv)
{
  graph G(argv[1]);
  G.parseGraph();

  bool printAns =false;
  if(argc>2)
    printAns=true;

  int V = G.num_nodes();
//---------------------------------------//
  printf("#nodes:%d\n",V);
//-------------------------------------//
 int E = G.num_edges();

 //---------------------------------------//
  printf("#edges:%d\n",E);
//-------------------------------------//

  int* edge_weight = G.getEdgeLen();

  //~ int* dist;

  int src=0;

  int *offset_array;
  int *edge_list;
  int *cpu_edge_weight;


   offset_array = (int *)malloc( (V+1)*sizeof(int));
   edge_list = (int *)malloc( (E)*sizeof(int));
   cpu_edge_weight = (int *)malloc( (E)*sizeof(int));
   //~ dist = (int *)malloc( (V)*sizeof(int));

  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    offset_array[i] = temp;
  }

  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edge_list[i] = temp;
    temp = edge_weight[i];
    cpu_edge_weight[i] = temp;
  }

  //~ for(int i=0; i< E; i++) {
    //~ int temp = edge_weight[i];
    //~ cpu_edge_weight[i] = temp;
  //~ }


    //~ cudaEvent_t start, stop;
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);
    //~ float milliseconds = 0;
    //~ cudaEventRecord(start,0);

	// replicanode *arr[V];

    SSSP(offset_array,edge_list, cpu_edge_weight ,src, V,E,printAns);
    //~ cudaDeviceSynchronize();

    //~ cudaEventRecord(stop,0);
    //~ cudaEventSynchronize(stop);
    //~ cudaEventElapsedTime(&milliseconds, start, stop);
    //~ printf("Time taken by function to execute is: %.6f ms\n", milliseconds);


  return 0;

}

