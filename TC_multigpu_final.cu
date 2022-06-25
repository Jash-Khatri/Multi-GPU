#include<stdio.h>
#include<stdlib.h>
#include<limits.h>
#include<cmath>
#include<algorithm>
#include<cuda.h>
#include"graph.hpp"


//__device__ int triangle_count = 0;

  __device__ bool check_if_nbr(int s, int d, int * gpu_OA, int *gpu_edgeList)   //we can move this to graph.hpp file
    {
      int startEdge=gpu_OA[s];
      int endEdge=gpu_OA[s+1]-1;

      if(gpu_edgeList[startEdge]==d)
          return true;
      if(gpu_edgeList[endEdge]==d)
         return true;   

       int mid = (startEdge+endEdge)/2;

      while(startEdge<=endEdge)
        {
       
          if(gpu_edgeList[mid]==d)
             return true;

          if(d<gpu_edgeList[mid])
             endEdge=mid-1;
          else
            startEdge=mid+1;   
          
          mid = (startEdge+endEdge)/2;

        }
      
      return false;

    }


__global__ void kernel(int * gpu_OA, int * gpu_edgeList, int V, int E, int *triangle_count, int dnum, int perdevicevertices, int devicecount, int minid) 
{

    unsigned int id = threadIdx.x + (blockDim.x * blockIdx.x);
    //id = id - dnum * perdevicevertices; 		// (i+1);     

   if( id >= (dnum*perdevicevertices - dnum*minid ) && id < ( (dnum == devicecount-1) ? (V - dnum*minid) : ( (dnum+1)*perdevicevertices ) ) )
   //if(  id < ( (dnum == devicecount-1) ? perdevicevertices+lastleftvertices : perdevicevertices ) )
   {
     // printf("id = %d",id);
      
    for (int edge = gpu_OA[id]; edge < gpu_OA[id+1]; edge++) 
    { 
      int u =  gpu_edgeList[edge] ;
      if (u < id )
      {
        for (int edge = gpu_OA[id]; edge <  gpu_OA[id+1]; edge ++) 
         { 
          int w = gpu_edgeList[edge] ;
          if (w > id )
          {
            if (check_if_nbr(u, w,gpu_OA,gpu_edgeList ) )
            {
              atomicAdd(triangle_count ,1);
            }
          }
        }
      }
    }


  }
    //printf("TC = %d",triangle_count);
}


void Compute_TC(int * OA, int * edgeList, int V, int E)
{
  
  // Somewhere on CPU
  int devicecount;
  cudaGetDeviceCount(&devicecount);
  std::cout << devicecount << "\n"; 

 // printf("hi from function\n");
   
   int *gpu_edgeList[devicecount];		
   int *gpu_OA[devicecount];	

  // creating the count triangle var for each GPU
  int *triangle_count[devicecount];  		

  // printf("V inside fun =%d",V);
  
  int perdevicevertices;
  int lastleftvertices;
  perdevicevertices = V/ devicecount ;
  lastleftvertices = V % devicecount;


  // traversing through the second half
  int minid = V+1;
		
  for( int id = perdevicevertices ; id < V ; id++ )
  {
	for (int edge = OA[id]; edge < OA[id+1]; edge++) 
	{
		minid = min(minid, edgeList[edge] );
	}
  }
  minid = min(minid,perdevicevertices);	
  std::cout << minid << "\n";

  for(int i=0;i<devicecount;i++)		// only allocate V/devicecount of space
  {	
	if( i == 0 )
	{
	cudaSetDevice(i);
  	cudaMalloc( &gpu_OA[i], sizeof(int) * ( 1 + perdevicevertices ) );
	}
	else
	{
	cudaSetDevice(i);
  	cudaMalloc( &gpu_OA[i], sizeof(int) * ( 1 + (V - minid) ) ) ;
	}
  }

  for(int i=0;i<devicecount;i++)		// only allocate the necessary edges
  {	
	if( i == 0 )
	{
  		cudaSetDevice(i);
  		cudaMalloc( &gpu_edgeList[i], sizeof(int) * (  OA[(i+1)*perdevicevertices] - OA[i*perdevicevertices]  ) );
  	}
	else
	{
		cudaSetDevice(i);
  		cudaMalloc( &gpu_edgeList[i], sizeof(int) * (  OA[V] - OA[minid] ) );
	}
  }

  for(int i=0;i<devicecount;i++)		
  {	
  	cudaSetDevice(i);
  	cudaMalloc( &triangle_count[i], (1)*sizeof(int) );
  }
  
  unsigned int block_size;
  unsigned int num_blocks;
 
  if(V <= 1024)
	{
		block_size = V;
		num_blocks = 1;
	}
	else
	{
		block_size = 1024;
		num_blocks = ceil(((float)V) / block_size);
	}


  // need to change the offset indices
  int *newOA;	
  newOA = (int *)malloc( (V+1)*sizeof(int));
  long long int count_indices = 0;

  	for(int i=0;i<V+1;i++)
  	{
  	      if(i > 0)
			count_indices += OA[i] - OA[i-1];
		else
			count_indices = 0;
		
		newOA[i] = count_indices;
	
	        if( i == minid )
			count_indices = 0;
	}
  

  for(int i=0;i<devicecount;i++)	// only V/decivecount memcpys needed
  {
	if( i == 0 )
	{
  	cudaSetDevice(i);
  	cudaMemcpy(gpu_OA[i], OA , sizeof(int) * (1+perdevicevertices), cudaMemcpyHostToDevice);
  	}
	else
	{
	cudaSetDevice(i);
  	cudaMemcpy(gpu_OA[i], newOA+minid, sizeof(int) * ( 1 + (V - minid) ), cudaMemcpyHostToDevice);	
	}
  }

  for(int i=0;i<devicecount;i++)
  {
	cudaSetDevice(i);
	cudaMemset(gpu_OA[i], 0, sizeof(int) );
  }

  int *newedgeList;
  newedgeList = (int *)malloc( (E)*sizeof(int));

  for(int i=0;i<E;i++)
  {
	newedgeList[i] = edgeList[i];
  }
  
  for(int j=OA[minid];j<OA[V];j++) 
  {
	newedgeList[j] -= (minid);
  }

  long long int sum_of_edges = 0;
  for(int i=0;i<devicecount;i++)	// only memcpy the necessary edges
  {
	if( i == 0 )
	{
  	cudaSetDevice(i);
  	cudaMemcpy(gpu_edgeList[i], edgeList + sum_of_edges, sizeof(int) * (  OA[(i+1)*perdevicevertices] - OA[i*perdevicevertices]  ), cudaMemcpyHostToDevice);
	sum_of_edges += (  OA[(i+1)*perdevicevertices] - OA[i*perdevicevertices] );
  	}
	else
	{
  	cudaSetDevice(i);
  	cudaMemcpy(gpu_edgeList[i], newedgeList + OA[minid] , sizeof(int) * ( OA[V] - OA[minid] ), cudaMemcpyHostToDevice);
	}
  }


  for(int i=0;i<devicecount;i++)	
  {
  cudaSetDevice(i);
  cudaMemset(triangle_count[i], 0, sizeof(int) );
  }  

  cudaEvent_t start, stop; ///TIMER START
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float milliseconds = 0;
  cudaEventRecord(start,0);

  printf("before kernel\n");
  
  // kernel call on each of the GPU -- Processing their chunck of vertices independently
  for(int i=0;i<devicecount;i++)		
  {
	cudaSetDevice(i);
  	kernel<<<num_blocks,block_size>>>(gpu_OA[i], gpu_edgeList[i], V, E, triangle_count[i], i, perdevicevertices, devicecount, minid );
  }

  for(int i=0;i<devicecount;i++)		
  {
	cudaSetDevice(i);
  	cudaDeviceSynchronize();
  }

  printf("after kernel\n");

  //STOP TIMER
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("GPU Time: %.6f ms \n", milliseconds);

  long long int sum = 0;
  for(int i=0;i<devicecount;i++)		
  {
	cudaSetDevice(i);
  	int count;
  	cudaMemcpy(&count, triangle_count[i], sizeof(int), cudaMemcpyDeviceToHost );
	 std::cout << count << "\n";
	sum += count;
  }	
	  
  printf("TC = %lld\n",sum);
 }



 int main(int argc , char ** argv)
{

  graph G(argv[1]);  //this will be changed
  G.parseGraph();
  
  int V = G.num_nodes();
  
 // printf("number pf nodes =%d",V);
  
  int E = G.num_edges();
  
//  printf("number pf edges =%d",E);
  

  int *OA;
  int *edgeList;
  
  
   OA = (int *)malloc( (V+1)*sizeof(int));
   edgeList = (int *)malloc( (E)*sizeof(int));
  
    
  for(int i=0; i<= V; i++) {
    int temp = G.indexofNodes[i];
    OA[i] = temp;
  }
  
  for(int i=0; i< E; i++) {
    int temp = G.edgeList[i];
    edgeList[i] = temp;
  }
  
 
  
  Compute_TC(OA, edgeList, V,E);

}
