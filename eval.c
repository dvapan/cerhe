// -*- compile-command: "gcc -g -Wall -o eval eval.c" -*-
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct vars_params{
     double start;
     double finish;
     int step_nums;
};

struct slice_coord{
     int a,b;
};

int main()
{
     int vars_count, poly_degree;
     scanf("%d %d", &vars_count, &poly_degree);
     struct vars_params vp[vars_count];
     for(int i = 0; i < vars_count; i++){
          scanf("%lf %lf %d",
                &vp[i].start,
                &vp[i].finish,
                &vp[i].step_nums);
     }
     
     double *vars[vars_count];
     for (int i = 0; i < vars_count; i++){
          vars[i] = malloc(sizeof(double) * vp[i].step_nums);
          double h = (vp[i].finish - vp[i].start)/(vp[i].step_nums - 1);
          for (int j = 0; j < vp[i].step_nums; j++)
               vars[i][j] = vp[i].start + j*h;
     }

     int reg_count;
     scanf("%d", &reg_count);

     struct slice_coord sl[reg_count][vars_count];
     
     for(int i = 0; i < reg_count; i++)
          for(int j = 0; j < vars_count; j++){
               scanf("%d %d",&sl[i][j].a, &sl[i][j].b);
          }

     for(int i = 0; i < reg_count; i++)
          for(int j = 0; j < vars_count; j++){
               printf("%2d %2d: ",sl[i][j].a, sl[i][j].b);
               for (int idx = sl[i][j].a; idx < sl[i][j].b-1; idx++)
                    printf("% 10.5lf ",vars[j][idx]);
               printf("% 8.5lf\n", vars[j][sl[i][j].b-1]);
          }
     
     return 0;
}
