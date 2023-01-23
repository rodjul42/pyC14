

int POP8_h_exp_rhs(int nrows, int ncols, double* indata, int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n );


int POP8_d_exp_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double p2nS, double p4nS , double pMonoSD  );

int  POP8P_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   );
   
int  POP8P_H2b_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double da, double db, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   );



int  POP8P_D_h_exp_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double k4n2x4n, double k8n2x8n, double k16n2x16n,
    double b4n, double b8n, double b16n, double b2x2n, double b2x4n, double b2x8n, double b2x16n,
    double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n   );


int  POP8P_D_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double k4n2x4n, double k8n2x8n, double k16n2x16n,
    double b4n, double b8n, double b16n, double b2x2n, double b2x4n, double b2x8n, double b2x16n,
    double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   );

