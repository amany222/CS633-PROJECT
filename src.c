#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

int sub_nz;
int sub_ny;
int sub_nx;
int rank, size, PX, PY, PZ, NX, NY, NZ, NC;


int valid_glb(int x,int y,int z){
  if(x<0 || x >=NX){
    return 0;
  }
  if(y<0 || y >=NY){
    return 0;
  }
  if(z<0 || z >=NZ){
    return 0;
  }
  return 1;
}

float min(float x,float y){
  if(x>y){
    return y;
  }
  return x;
}
float max(float x,float y){
  if(x<y){
    return y;
  }
  return x;
}

int get(int nx,int ny,int nz){
  if(nx < 0 || nx >=sub_nx){
    return -1;
  }
  if(ny < 0 || ny >=sub_ny){
    return -1;
  }
  if(nz < 0 || nz >=sub_nz){
    return -1;
  }
  int rank = nz*(PY*PX) + ny*PX + nx;
  if(rank >= 0 && rank < size){
    return rank;
  }
  return -1;
}


int numb_request = 0;
MPI_Request reqs[12];
MPI_Status stats[12];


int main(int argc, char **argv) {
  char input_file[256], output_file[256];
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 10) {
    if (rank == 0) {
      printf("Usage: mpirun -np P ./executable <input_file> PX PY PZ NX NY NZ NC <output_file>\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  sscanf(argv[1], "%s", input_file);
  PX = atoi(argv[2]);
  PY = atoi(argv[3]);
  PZ = atoi(argv[4]);
  NX = atoi(argv[5]);
  NY = atoi(argv[6]);
  NZ = atoi(argv[7]);
  NC = atoi(argv[8]);
  sscanf(argv[9], "%s", output_file);

  if (size != PX * PY * PZ) {
    if (rank == 0) {
      printf("Error: The number of processes must be equal to PX * PY * PZ.\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int x_rank = rank % PX;
  int y_rank = (rank / PX) % PY;
  int z_rank = rank / (PX * PY);

  sub_nx = NX / PX;
  sub_ny = NY / PY;
  sub_nz = NZ / PZ;

  double time1 = MPI_Wtime();

    float *data4 = (float *)malloc(sub_nz* sub_ny * sub_nx * NC * sizeof(float));
    
    int off = z_rank*sub_nz*NY*NX*NC + y_rank*sub_ny*NX*NC + x_rank*sub_nx*NC;

    MPI_File fh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    for(int k=0;k < sub_nz;k++){
        for(int j=0;j<sub_ny;j++){
            MPI_Offset offset = (MPI_Offset) off * sizeof(float);
            MPI_File_read_at(fh, offset, &data4[k*sub_ny*sub_nx*NC + j*sub_nx*NC], NC*sub_nx,MPI_FLOAT, &status);
            off += NX*NC;
        }
        off += (NY-sub_ny)*NX*NC;
    }

  double time2 = MPI_Wtime();

  float lcl_mn[NC], lcl_mx[NC];
  int mn_count[NC],mx_count[NC];

  for(int i = 0;i<NC;i++){
    lcl_mn[i] = FLT_MAX;
    lcl_mx[i] = FLT_MIN;
    mn_count[i] = 0;
    mx_count[i] = 0;
  }


  float recv_right[sub_nz][sub_ny][NC];
  float recv_left[sub_nz][sub_ny][NC];
  float recv_bottom[sub_ny][sub_nx][NC];
  float recv_top[sub_ny][sub_nx][NC];
  float recv_front[sub_nz][sub_nx][NC];
  float recv_back[sub_nz][sub_nx][NC];
    

  MPI_Datatype y_dir;
  MPI_Type_vector(sub_nz, NC * sub_nx, NC * sub_nx * sub_ny, MPI_FLOAT, &y_dir);
  MPI_Type_commit(&y_dir);
  
  if (y_rank > 0) {
      MPI_Isend(data4, 1, y_dir, get(x_rank, y_rank - 1, z_rank), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
      MPI_Irecv(recv_front, NC * sub_nz * sub_nx, MPI_FLOAT, get(x_rank, y_rank - 1, z_rank), get(x_rank, y_rank - 1, z_rank), MPI_COMM_WORLD, &reqs[numb_request++]);
  }
  if (y_rank < PY - 1) {
      MPI_Isend(&data4[NC * sub_nx * (sub_ny - 1)], 1, y_dir, get(x_rank, y_rank + 1, z_rank), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
      MPI_Irecv(recv_back, NC * sub_nz * sub_nx, MPI_FLOAT, get(x_rank, y_rank + 1, z_rank), get(x_rank, y_rank + 1, z_rank), MPI_COMM_WORLD, &reqs[numb_request++]);
  }
  MPI_Type_free(&y_dir);
  


  MPI_Datatype x_dir;
  MPI_Type_vector(sub_nz * sub_ny, NC, NC * sub_nx, MPI_FLOAT, &x_dir);
  MPI_Type_commit(&x_dir);
  
    if (x_rank > 0) {
      MPI_Isend(data4, 1, x_dir, get(x_rank - 1, y_rank, z_rank), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
      MPI_Irecv(recv_left, NC * sub_nz * sub_ny, MPI_FLOAT, get(x_rank - 1, y_rank, z_rank), get(x_rank - 1, y_rank, z_rank), MPI_COMM_WORLD, &reqs[numb_request++]);
    }
    if (x_rank < PX - 1) {
        MPI_Isend(&data4[NC * (sub_nx - 1)], 1, x_dir, get(x_rank + 1, y_rank, z_rank), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
        MPI_Irecv(recv_right, NC * sub_nz * sub_ny, MPI_FLOAT, get(x_rank + 1, y_rank, z_rank), get(x_rank + 1, y_rank, z_rank), MPI_COMM_WORLD, &reqs[numb_request++]);
    }
    MPI_Type_free(&x_dir);

    if (z_rank > 0) {
        MPI_Isend(data4, NC*sub_nx*sub_ny, MPI_FLOAT, get(x_rank, y_rank, z_rank-1), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
        MPI_Irecv(recv_top, NC*sub_nx*sub_ny, MPI_FLOAT, get(x_rank, y_rank, z_rank-1), get(x_rank, y_rank, z_rank-1), MPI_COMM_WORLD, &reqs[numb_request++]);
    }
    if (z_rank < PX - 1) {
        MPI_Isend(&data4[NC * (sub_nz - 1)*sub_nx*sub_ny], NC*sub_nx*sub_ny, MPI_FLOAT, get(x_rank , y_rank, z_rank+1), rank, MPI_COMM_WORLD, &reqs[numb_request++]);
        MPI_Irecv(recv_bottom, NC*sub_nx*sub_ny, MPI_FLOAT, get(x_rank , y_rank, z_rank+1), get(x_rank , y_rank, z_rank+1), MPI_COMM_WORLD, &reqs[numb_request++]);
    }

    
  MPI_Waitall(numb_request,reqs,stats);

  int m[3] = {-1,0,1};


  float *dat = (float *)malloc((sub_nz+2)*(sub_ny+2)*(sub_nx+2)*NC*sizeof(float));

  for(int i=1;i<=sub_nx;i++){
    for(int j =1;j<=sub_ny;j++){
        for(int k=1;k<=sub_nz;k++){
            for(int x=0;x<NC;x++){
                // dat[k][j][i][x] = data4[k-1][j-1][i-1][x];
                // dat[k][j][i][x] = data4[(k-1)*sub_ny*sub_nx*NC + (j-1)*sub_nx*NC + (i-1)*NC + x];
                dat[k*(sub_ny+2)*(sub_nx+2)*NC + (j)*(sub_nx+2)*NC + (i)*NC + x] = data4[(k-1)*sub_ny*sub_nx*NC + (j-1)*sub_nx*NC + (i-1)*NC + x];
            }
        }
    }
  }

  free(data4);


    for(int i=1;i<=sub_nx;i++){
        for(int j=1;j<=sub_ny;j++){
            for(int x=0;x<NC;x++){
                // dat[0][j][i][x] = recv_top[j-1][i-1][x];
                // dat[sub_nz+1][j][i][x] = recv_bottom[j-1][i-1][x];
                dat[0*(sub_ny+2)*(sub_nx+2)*NC + j*(sub_nx+2)*NC + i*NC + x] = recv_top[j-1][i-1][x];
                dat[(sub_nz+1)*(sub_ny+2)*(sub_nx+2)*NC + j*(sub_nx+2)*NC + i*NC + x] = recv_bottom[j-1][i-1][x];
                // dat[k][0][i][x] = recv_front[k-1][i-1][x];
                // dat[k][sub_ny+1][i][x] = recv_back[k-1][i-1][x];

            }
        }
    }
    
    for(int i=1;i<=sub_nz;i++){
        for(int j=1;j<=sub_ny;j++){
            for(int x=0;x<NC;x++){
                // dat[i][j][0][x] = recv_left[i-1][j-1][x];
                // dat[i][j][sub_nx+1][x] = recv_right[i-1][j-1][x];
                dat[i*(sub_ny+2)*(sub_nx+2)*NC + j*(sub_nx+2)*NC + 0*NC + x] = recv_left[i-1][j-1][x];
                dat[i*(sub_ny+2)*(sub_nx+2)*NC + j*(sub_nx+2)*NC + (sub_nx+1)*NC + x] = recv_right[i-1][j-1][x];
            }
        }
    }
    for(int i=1;i<=sub_nx;i++){
        for(int j=1;j<=sub_nz;j++){
            for(int x=0;x<NC;x++){
                // dat[j][0][i][x] = recv_front[j-1][i-1][x];
                // dat[j][sub_ny+1][i][x] = recv_back[j-1][i-1][x];
                dat[j*(sub_ny+2)*(sub_nx+2)*NC + 0*(sub_nx+2)*NC + i*NC + x] = recv_front[j-1][i-1][x];
                dat[j*(sub_ny+2)*(sub_nx+2)*NC + (sub_ny+1)*(sub_nx+2)*NC + i*NC + x] = recv_back[j-1][i-1][x];
            }
        }
    }

    for(int i=1;i<=sub_nx;i++){
        for(int j=1;j<=sub_ny;j++){
            for(int k=1;k<=sub_nz;k++){
                for(int x=0;x<NC;x++){
                    int mn = 1;
                    int mx = 1;
                    int ind1 = k*(sub_ny+2)*(sub_nx+2)*NC + j*(sub_nx+2)*NC + i*NC + x;
                    for(int m1 = -1;m1<=1;m1++){
                        if(m1 == 0){
                            continue;
                        }
                        int ni = i+m1;
                        int nj = j;
                        int nk = k;
                        
                        int glb_ni = (sub_nx*x_rank) + ni -1;
                        int glb_nj = (sub_ny*y_rank) + nj -1;
                        int glb_nk = (sub_nz*z_rank) + nk -1;
                        
                        int ind2 = nk*(sub_ny+2)*(sub_nx+2)*NC + nj*(sub_nx+2)*NC + ni*NC + x;

                        if(valid_glb(glb_ni,glb_nj,glb_nk)){
                            if(dat[ind1] <= dat[ind2]){
                                mx = 0;
                            }
                            if(dat[ind1] >= dat[ind2]){
                                mn = 0;
                            }
                        }
                    }
                    for(int m1 = -1;m1<=1;m1++){
                        if(m1 == 0){
                            continue;
                        }
                        int ni = i;
                        int nj = j+m1;
                        int nk = k;
                        
                        int glb_ni = (sub_nx*x_rank) + ni -1;
                        int glb_nj = (sub_ny*y_rank) + nj -1;
                        int glb_nk = (sub_nz*z_rank) + nk -1;
                        
                        int ind2 = nk*(sub_ny+2)*(sub_nx+2)*NC + nj*(sub_nx+2)*NC + ni*NC + x;

                        if(valid_glb(glb_ni,glb_nj,glb_nk)){
                            if(dat[ind1] <= dat[ind2]){
                                mx = 0;
                            }
                            if(dat[ind1] >= dat[ind2]){
                                mn = 0;
                            }
                        }
                    }
                    for(int m1 = -1;m1<=1;m1++){
                        if(m1 == 0){
                            continue;
                        }
                        int ni = i;
                        int nj = j;
                        int nk = k+m1;
                        
                        int glb_ni = (sub_nx*x_rank) + ni -1;
                        int glb_nj = (sub_ny*y_rank) + nj -1;
                        int glb_nk = (sub_nz*z_rank) + nk -1;
                        
                        int ind2 = nk*(sub_ny+2)*(sub_nx+2)*NC + nj*(sub_nx+2)*NC + ni*NC + x;

                        if(valid_glb(glb_ni,glb_nj,glb_nk)){
                            if(dat[ind1] <= dat[ind2]){
                                mx = 0;
                            }
                            if(dat[ind1] >= dat[ind2]){
                                mn = 0;
                            }
                        }
                    }
                    if(mn == 1){
                        mn_count[x]++;
                        lcl_mn[x] = min(lcl_mn[x],dat[ind1]);
                    }
                    if(mx == 1){
                        mx_count[x]++;
                        lcl_mx[x] = max(lcl_mx[x],dat[ind1]);
                    }
                }
            }
        }
    }

    free(dat);


  float glb_mn[NC],glb_mx[NC];
  int glb_mn_count[NC],glb_mx_count[NC];

  MPI_Reduce(lcl_mn,glb_mn,NC,MPI_FLOAT,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(lcl_mx,glb_mx,NC,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(mn_count,glb_mn_count,NC,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(mx_count,glb_mx_count,NC,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);

  double time3 = MPI_Wtime();


  if (rank == 0) {
    FILE *file = fopen(output_file, "w");
    if (file == NULL) {
      printf("Error opening file for writing!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int i = 0;i<NC;i++){
      fprintf(file,"( %d , %d )",glb_mn_count[i],glb_mx_count[i]);
      if(i<NC-1){
        fprintf(file," , ");
      }else{
        fprintf(file,"\n");
      }
    }
    for(int i = 0;i<NC;i++){
      fprintf(file,"( %f , %f )",glb_mn[i],glb_mx[i]);
      if(i<NC-1){
        fprintf(file," , ");
      }else{
        fprintf(file,"\n");
      }
    }
    fclose(file);
  }
  double time4 = MPI_Wtime();
  double t1 = time2-time1;
  double t2 = time3-time2;
  double t3 = time4-time1;

  double max_t1, max_t2, max_t3;
  MPI_Reduce(&t1, &max_t1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t2, &max_t2, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&t3, &max_t3, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


  if (rank == 0) {
    FILE *file = fopen(output_file, "a");
    if (file == NULL) {
      printf("Error opening file for writing!\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    fprintf(file,"%f %f %f ",max_t1,max_t2,max_t3); 
    fclose(file);
  }


  MPI_Finalize();
  return 0;
}