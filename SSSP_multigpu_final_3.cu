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
  bool * gpu_finished, 			// updated
  int perdevicevertices,
  int devicecount ,
  int minid)
{
  //unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
  unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);	// changes here
  unsigned int v = id + dnum*minid;
  //if ( id >= (dnum*perdevicevertices ) && id < ( (dnum+1)*perdevicevertices )  )
  if( id < ( (dnum == devicecount-1) ? ( V - dnum*minid ) : ( (dnum+1)*perdevicevertices ) ) )
  {
   if (gpu_modified_prev[id + dnum*minid] ){		// updated

//	printf( "%d " , id );
      for (int edge = gpu_offset_array[ id ]; edge < gpu_offset_array[ id + 1 ]; edge ++)
      {
        int nbr = gpu_edge_list[edge] ;
	nbr = nbr + dnum*minid;				// updated
        int e = edge;
        int dist_new;
        if(gpu_dist[id + dnum*minid] != INT_MAX)	// updated
          dist_new = gpu_dist[v] + gpu_weight[e];

	 if (gpu_dist[nbr] > dist_new)
         {
           atomicMin(&gpu_dist[nbr] , dist_new);
           gpu_modified_next[nbr]=true;
           *gpu_finished = false ;
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
  std::cout << "Total num of GPUS: " << devicecount << "\n";

  int perdevicevertices;
  int lastleftvertices;
  perdevicevertices = V / devicecount ;
  lastleftvertices = V % devicecount;

  // traversing through the second half
  int minid = V+1;
		
  for( int id = perdevicevertices ; id < V ; id++ )
  {
	for (int edge = offset_array[id]; edge < offset_array[id+1]; edge++) 
	{
		minid = min(minid, edge_list[edge] );
	}
  }
  minid = min(minid,perdevicevertices);	
  std::cout << minid << "\n";

  //CSR VARS
  int * gpu_offset_array[devicecount];
  for(int i=0;i<devicecount;i++)
  {	
	if( i == 0 )  	
	{
		cudaSetDevice(i);
	        cudaMalloc(&gpu_offset_array[i],sizeof(int) * (1+perdevicevertices) );
	}
  	else
	{
		cudaSetDevice(i);
 		cudaMalloc(&gpu_offset_array[i],sizeof(int) *( 1 + (V - minid) ) );
	}
  }

	//std::cout << "Done" << "\n";

  int * gpu_edge_list[devicecount];		
  for(int i=0;i<devicecount;i++)
  {
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMalloc(&gpu_edge_list[i],sizeof(int) *(  offset_array[(i+1)*perdevicevertices] - offset_array[i*perdevicevertices] ) );
 	}
	else
	{
		cudaSetDevice(i);
                cudaMalloc(&gpu_edge_list[i],sizeof(int) * (  offset_array[V] - offset_array[minid] )  );
	}
  }
	
  int * gpu_edge_weight[devicecount];		// new Add
  for(int i=0;i<devicecount;i++)
  {
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMalloc(&gpu_edge_weight[i],sizeof(int) *(  offset_array[(i+1)*perdevicevertices] - offset_array[i*perdevicevertices] )  );
	}
	else
	{
		cudaSetDevice(i);
                cudaMalloc(&gpu_edge_weight[i],sizeof(int) * (  offset_array[V] - offset_array[minid] ) );
	}
  }

	//std::cout << "Done" << "\n";

  // RESUT VAR
  int *gpu_dist[devicecount];		// new Add
  
  //int dist[devicecount][V];
  int **dist;
  dist = new int*[devicecount];
  for(int i=0; i<devicecount; i++)
    dist[i] = new int[V];

  for(int i=0;i<devicecount;i++)	// new Add
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_dist[i],sizeof(int) *(V));
  }
	//std::cout << "Done" << "\n";

  // EXTRA VARS

  //bool cpu_modified_prev[devicecount][V];
  bool **cpu_modified_prev;			// new added
  cpu_modified_prev = new bool*[devicecount];
  for(int i=0; i<devicecount; i++)
    cpu_modified_prev[i] = new bool[V];

  bool *gpu_modified_prev[devicecount];		// new added
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_modified_prev[i],sizeof(bool) *(V));
  }

  bool * gpu_modified_next[devicecount];	// new added
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_modified_next[i],sizeof(bool) *(V));
  }

  bool * gpu_finished[devicecount];		// new added
  for(int i=0;i<devicecount;i++)
  {
  cudaSetDevice(i);
  cudaMalloc(&gpu_finished[i],sizeof(bool) *(1));
  }
  //printf("Allocation finished\n");

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
  
  for(int i=0;i<devicecount;i++)		// new added
  {
  cudaSetDevice(i);
  initKernel2<int,bool> <<<num_blocks,block_size>>>(V,gpu_dist[i], INT_MAX,gpu_modified_prev[i], false);
  }

  // This comes from DSL. Single thread kernel
  for(int i=0;i<devicecount;i++)		// new added
  {
  cudaSetDevice(i);
  initKernel0<int> <<<1,1>>>(gpu_dist[i], src,0);
  }

  for(int i=0;i<devicecount;i++)		// new added
  {
  cudaSetDevice(i);
  initKernel0<bool> <<<1,1>>>(gpu_modified_prev[i], src,true);
  }

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  // need to change the offset indices
  int *newOA;	
  newOA = (int *)malloc( (V+1)*sizeof(int));
  long long int count_indices = 0;

  	for(int i=0;i<V+1;i++)
  	{
  	      if(i > 0)
			count_indices += offset_array[i] - offset_array[i-1];
		else
			count_indices = 0;
		
		newOA[i] = count_indices;
	
	        if( i == minid )
			count_indices = 0;
	}


  // CSR
  for(int i=0;i<devicecount;i++)
  {
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMemcpy (gpu_offset_array[i]  , offset_array    , sizeof(int)   * (1+perdevicevertices) , cudaMemcpyHostToDevice);
  	}
	else
	{
		cudaSetDevice(i);
                cudaMemcpy (gpu_offset_array[i]  , newOA+minid    , sizeof(int)   * ( 1 + (V - minid) ) , cudaMemcpyHostToDevice);
	}
  }

  for(int i=0;i<devicecount;i++)
  {
	cudaSetDevice(i);
	cudaMemset( gpu_offset_array[i], 0, sizeof(int) );
  }

  int *newedgeList;
  newedgeList = (int *)malloc( (E)*sizeof(int));

  for(int i=0;i<E;i++)
  {
	newedgeList[i] = edge_list[i];
  }
  
  for(int j= offset_array[minid];j<offset_array[V];j++) 
  {
	newedgeList[j] -= (minid);
  }


  for(int i=0;i<devicecount;i++)
  {
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMemcpy (gpu_edge_list[i]     , edge_list       , sizeof(int)   * (  offset_array[(i+1)*perdevicevertices] - offset_array[i*perdevicevertices]  )  , cudaMemcpyHostToDevice);
 	}
	else
	{
		cudaSetDevice(i);
                cudaMemcpy (gpu_edge_list[i]     , newedgeList + offset_array[minid]    , sizeof(int) * ( offset_array[V] - offset_array[minid] )  , cudaMemcpyHostToDevice);
	}
  }

  for(int i=0;i<devicecount;i++)		// new added
  {
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMemcpy (gpu_edge_weight[i]   , cpu_edge_weight , sizeof(int)   * (  offset_array[(i+1)*perdevicevertices] - offset_array[i*perdevicevertices]  )  , cudaMemcpyHostToDevice);
 	}
	else
	{
		cudaSetDevice(i);
                cudaMemcpy (gpu_edge_weight[i]   , cpu_edge_weight + offset_array[minid] , sizeof(int) * ( offset_array[V] - offset_array[minid] )  , cudaMemcpyHostToDevice);

	}
  }

  // COMES FROM DSL so CPU VARIABLE
  bool* finished = new bool[1];
  *finished = false; // to kick start

  //for(int i=0;i<V+1;i++)
  //	printf( "%d ", offset_array[i] );
  //printf("\n");

  //for(int i=0;i<E;i++)
  //	printf( "%d ", edge_list[i] );
  //printf("\n");

  int k =0; /// We need it only for count iterations attached to FIXED pt
  while ( !(*finished) )
  {
    //~ finished[0]=true;   /// I guess  we do not need this line overwrritten in memcpy below
    for(int i=0;i<devicecount;i++)		// new added
    {
    cudaSetDevice(i);
    initKernel<bool> <<< 1, 1>>>(1, gpu_finished[i], true);
    }
	
    // kernel call on each of the GPU -- Processing their chunck of vertices independently
    for(int i=0;i<devicecount;i++)		// new added
    {
	cudaSetDevice(i);	
	//cudaSetDevice(0);
	Compute_SSSP_kernel<<<num_blocks , block_size>>>(i, gpu_offset_array[i], gpu_edge_list[i],  gpu_edge_weight[i] ,gpu_dist[i] ,V , gpu_modified_prev[i], gpu_modified_next[i], gpu_finished[i], perdevicevertices, devicecount, minid );
    }
	// No explicit CDS required for each device here as there are cudaMemcopies happening on each device later

	//printf("Hellooo\n");
    for(int i=0;i<devicecount;i++)		// new added
    {
        cudaSetDevice(i);
    	initKernel<bool><<<num_blocks,block_size>>>(V, gpu_modified_prev[i], false);
    }
        //printf("Hellooo\n");
    
    bool cpuarr_finished[devicecount];
    bool* currvalue;
    currvalue = (bool *)malloc(sizeof(bool) *(1));

   // printf("Hellooo\n");
    
    for(int i=0;i<devicecount;i++)		// new added
    {
	//printf("%d ", i);
        cudaSetDevice(i);
    	cudaMemcpy(currvalue, gpu_finished[i],  sizeof(bool) *(1), cudaMemcpyDeviceToHost);	
    	cpuarr_finished[i] = *currvalue;
    }

    //printf("Hellooo\n");

    *finished = true;
	
    for(int i=0;i<devicecount;i++)		// new added
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
    for(int i=0;i<devicecount;i++)		// new added
    {
        cudaSetDevice(i);
    	bool *tempModPtr  = gpu_modified_next[i];
    	gpu_modified_next[i] = gpu_modified_prev[i];
    	gpu_modified_prev[i] = tempModPtr;
    }

   // printf("I am in loop\n");
    
    for(int i=0;i<devicecount;i++)
    {
        cudaSetDevice(i);
        // copying the GPU computed dist gpu_dist to CPU for updation purpose
	cudaMemcpy(dist[i],gpu_dist[i] , sizeof(int) * (V), cudaMemcpyDeviceToHost);	
	cudaMemcpy(cpu_modified_prev[i],gpu_modified_prev[i] , sizeof(bool) * (V), cudaMemcpyDeviceToHost);			
    }
	
    
    // update the CPU replica_info with the latest dist compute from dist array
    
    for(int j=0;j<V;j++)
    {
	int min_dist = INT_MAX;
	for(int i=0;i<devicecount;i++)
	{
		min_dist = min(min_dist,dist[i][j]);
	}
	for(int i=0;i<devicecount;i++)
        {
                dist[i][j] = min_dist;
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

    if( k== V-1 )
	    break;

    if(k==V)    // NEED NOT GENERATE. DEBUG Only
    {
      std::cout<< "THIS SHOULD NEVER HAPPEN" << '\n';
      exit(0);
    }
     // printf("Hi\n");
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
    for(int i=0;i<1;i++)
    {
        cudaSetDevice(i);
    	cudaMemcpy(dist[i],gpu_dist[i] , sizeof(int) * (V), cudaMemcpyDeviceToHost);
    }

    for( int j=0;j<1; j++ )
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

