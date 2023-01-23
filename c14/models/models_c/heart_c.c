#include <math.h>

static const double TH_NULL = 1e-4;

inline unsigned int age_id(unsigned int ncols,unsigned int id){
    return ncols*0+id;
}
inline unsigned int Ni_2n(unsigned int ncols,unsigned int id){
    return ncols*1+id;
}
inline unsigned int Ni_4n(unsigned int ncols,unsigned int id){
    return ncols*2+id;
}
inline unsigned int Ni_8n(unsigned int ncols,unsigned int id){
    return ncols*3+id;
}
inline unsigned int age_d_id(unsigned int ncols,unsigned int id){
    return ncols*4+id;
}
inline unsigned int p2nSD_id(unsigned int ncols,unsigned int id){
    return ncols*5+id;
}
inline unsigned int p4nSD_id(unsigned int ncols,unsigned int id){
    return ncols*6+id;
}
inline unsigned int p8nSD_id(unsigned int ncols,unsigned int id){
    return ncols*7+id;
}
inline unsigned int p2nS_id(unsigned int ncols,unsigned int id){
    return ncols*8+id;
}
inline unsigned int p4nS_id(unsigned int ncols,unsigned int id){
    return ncols*9+id;
}

inline unsigned int age_l_id(unsigned int ncols,unsigned int id){
    return ncols*10+id;
}
inline unsigned int ICM_id(unsigned int ncols,unsigned int id){
    return ncols*10+id;
}


inline unsigned int p2n_id(unsigned int ncols,unsigned int id){
    return ncols*1+id;
}
inline unsigned int dtp2n_id(unsigned int ncols,unsigned int id){
    return ncols*2+id;
}
inline unsigned int p4n_id(unsigned int ncols,unsigned int id){
    return ncols*3+id;
}
inline unsigned int dtp4n_id(unsigned int ncols,unsigned int id){
    return ncols*4+id;
}
inline unsigned int p8n_id(unsigned int ncols,unsigned int id){
    return ncols*5+id;
}
inline unsigned int dtp8n_id(unsigned int ncols,unsigned int id){
    return ncols*6+id;
}
inline unsigned int pMono_id(unsigned int ncols,unsigned int id){
    return ncols*7+id;
}
inline unsigned int dtpMono_id(unsigned int ncols,unsigned int id){
    return ncols*8+id;
}


inline unsigned int c2n_id(unsigned int ncols,unsigned int id){
    return ncols*0+id;
}
inline unsigned int c4n_id(unsigned int ncols,unsigned int id){
    return ncols*1+id;
}
inline unsigned int c8n_id(unsigned int ncols,unsigned int id){
    return ncols*2+id;
}
inline unsigned int c16n_id(unsigned int ncols,unsigned int id){
    return ncols*3+id;
}
inline unsigned int c2x2n_id(unsigned int ncols,unsigned int id){
    return ncols*4+id;
}
inline unsigned int c2x4n_id(unsigned int ncols,unsigned int id){
    return ncols*5+id;
}
inline unsigned int c2x8n_id(unsigned int ncols,unsigned int id){
    return ncols*6+id;
}
inline unsigned int c2x16n_id(unsigned int ncols,unsigned int id){
    return ncols*7+id;
}


const double Ntotal = 1;
const double dtNtotal = 0;
const double EPS = 0.05;


void calc_populations_h(unsigned int id, int ncols, double* indata, double t, double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n ,
    double& p2n, double& dtp2n, double& p4n, double& dtp4n, double& p8n, double& dtp8n){
        p2n =    (n2_0-indata[Ni_2n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_2n(ncols,id)];
        dtp2n = -1*(n2_0-indata[Ni_2n(ncols,id)])*exp(-t/t0_2n)/t0_2n;

        p4n =    (n4_0-indata[Ni_4n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_4n(ncols,id)];
        dtp4n = -1*(n4_0-indata[Ni_4n(ncols,id)])*exp(-t/t0_2n)/t0_2n;


        p8n =    (n8_0-indata[Ni_8n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_8n(ncols,id)];
        dtp8n = -1*(n8_0-indata[Ni_8n(ncols,id)])*exp(-t/t0_2n)/t0_2n;
    }

void calc_populations_d(unsigned int id, int ncols, double* indata, double t, double p2nS, double p4nS, double n2_0 ,double n4_0 ,double n8_0 , double t0_2n, double pMonoS, double pMonoSD,
    double& p2n, double& dtp2n, double& p4n, double& dtp4n, double& p8n, double& dtp8n, double& pMono, double& dtpMono){ 
        
        if (t< indata[age_d_id(ncols,id)]){
            p2n =    (n2_0-indata[Ni_2n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_2n(ncols,id)];
            dtp2n = -1*(n2_0-indata[Ni_2n(ncols,id)])*exp(-t/t0_2n)/t0_2n;      
            
            p4n =    (n4_0-indata[Ni_4n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_4n(ncols,id)];
            dtp4n = -1*(n4_0-indata[Ni_4n(ncols,id)])*exp(-t/t0_2n)/t0_2n;

            p8n =    (n8_0-indata[Ni_8n(ncols,id)])*exp(-t/t0_2n)+indata[Ni_8n(ncols,id)];
            dtp8n = -1*(n8_0-indata[Ni_8n(ncols,id)])*exp(-t/t0_2n)/t0_2n;
            
            pMono = pMonoS;
            dtpMono = 0;
        }else{
            double p8nS = 1 - p2nS - p4nS;
            double TMP = (t-indata[age_d_id(ncols,id)])/(indata[age_id(ncols,id)] - indata[age_d_id(ncols,id)]);
            p2n = p2nS - TMP * (p2nS - indata[p2nSD_id(ncols,id)]); 
            p4n = p4nS - TMP * (p4nS - indata[p4nSD_id(ncols,id)]); 
            p8n = p8nS - TMP * (p8nS - indata[p8nSD_id(ncols,id)]);

            pMono = pMonoS -  TMP * (pMonoS - pMonoSD);
            
            
            TMP = -1/(indata[age_id(ncols,id)] - indata[age_d_id(ncols,id)]);
            dtp2n =  TMP * (p2nS - indata[p2nSD_id(ncols,id)]); 
            dtp4n =  TMP * (p4nS - indata[p4nSD_id(ncols,id)]); 
            dtp8n =  TMP * (p8nS - indata[p8nSD_id(ncols,id)]); 
            
            dtpMono = TMP * (pMonoS - pMonoSD);

        }
    }



int rhs(int ncols, int id, double* Catm, double* old_data, double* new_data, double d,
        double p2n, double dtp2n, double p4n, double dtp4n, double p8n, double dtp8n, double pMono, double dtpMono ){
        double N2n = Ntotal*p2n*pMono;
        double N4n = Ntotal*p4n*pMono;
        double N8n = Ntotal*p8n*pMono;
        double N16nb = -(Ntotal*(-1 + p2n + p4n + p8n)*pMono);
        double N2x2n = -(Ntotal*p2n*(-1 + pMono));
        double N2x4n = -(Ntotal*p4n*(-1 + pMono));
        double N2x8n = -(Ntotal*p8n*(-1 + pMono));
        double N2x16nb = Ntotal*(-1 + p2n + p4n + p8n)*(-1 + pMono);

        double N16n   = N16nb  <TH_NULL ? TH_NULL : N16nb;
        double N2x16n = N2x16nb<TH_NULL ? TH_NULL : N2x16nb;


        double dtN2n = Ntotal*pMono*dtp2n + p2n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN4n = Ntotal*pMono*dtp4n + p4n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN8n = Ntotal*pMono*dtp8n + p8n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN16n = -(pMono*((-1 + p2n + p4n + p8n)*dtNtotal + Ntotal*(dtp2n + dtp4n + dtp8n))) - Ntotal*(-1 + p2n + p4n + p8n)*dtpMono;
        double dtN2x2n = -((-1 + pMono)*(p2n*dtNtotal + Ntotal*dtp2n)) - Ntotal*p2n*dtpMono;
        double dtN2x4n = -((-1 + pMono)*(p4n*dtNtotal + Ntotal*dtp4n)) - Ntotal*p4n*dtpMono;
        double dtN2x8n = -((-1 + pMono)*(p8n*dtNtotal + Ntotal*dtp8n)) - Ntotal*p8n*dtpMono;
        double dtN2x16n = (-1 + p2n + p4n + p8n)*(-1 + pMono)*dtNtotal + Ntotal*((-1 + pMono)*(dtp2n + dtp4n + dtp8n) + (-1 + p2n + p4n + p8n)*dtpMono);

        double b2n = (d*(N16n + N2n + N2x16n + N2x2n + N2x4n + N2x8n + N4n + N8n) + dtN16n + dtN2n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n + dtN4n + dtN8n)/N2n;
        double k2n4n = (d*(N16n + N4n + N8n) + dtN16n + dtN4n + dtN8n)/N2n;
        double k4n8n = (d*(N16n + N8n) + dtN16n + dtN8n)/N4n;
        double k8n16n = (d*N16n + dtN16n)/N8n;
        double k2n2x2n = (d*(N2x16n + N2x2n + N2x4n + N2x8n) + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n)/N2n;
        double k2x2n2x4n = (d*(N2x16n + N2x4n + N2x8n) + dtN2x16n + dtN2x4n + dtN2x8n)/N2x2n;
        double k2x4n2x8n = (d*(N2x16n + N2x8n) + dtN2x16n + dtN2x8n)/N2x4n;
        double k2x8n2x16n = (d*N2x16n + dtN2x16n)/N2x8n;

        
        k2x8n2x16n = N16nb  <TH_NULL ? 0 : k2x8n2x16n;
        if ( (b2n <0) || (k2n4n <0) || (k4n8n <0) || (k8n16n <0) || (k2n2x2n <0) || (k2x2n2x4n <0) || (k2x4n2x8n<0) || (k2x8n2x16n<0) ) {
            return 1;
        }     
        if ( (N2n <0) || (N4n <0) || (N8n <0) || (N2x2n <0) || (N2x4n <0) || (N2x8n <0) || (N16nb<-TH_NULL) || (N2x16nb<-TH_NULL) ) {
            return 1;
        }


        new_data[c2n_id(ncols,id)] = b2n*(-old_data[c2n_id(ncols,id)] + Catm[id]);
        new_data[c4n_id(ncols,id)] = (k2n4n*(old_data[c2n_id(ncols,id)] - 2*old_data[c4n_id(ncols,id)] +Catm[id])*N2n)/(2.*N4n);
        new_data[c8n_id(ncols,id)] = (k4n8n*(old_data[c4n_id(ncols,id)] - 2*old_data[c8n_id(ncols,id)] + Catm[id])*N4n)/(2.*N8n);
        if (N16nb  <TH_NULL){
            new_data[c16n_id(ncols,id)] = new_data[c8n_id(ncols,id)];
        }else{
            new_data[c16n_id(ncols,id)] = (k8n16n*(-2*old_data[c16n_id(ncols,id)] + old_data[c8n_id(ncols,id)] + Catm[id])*N8n)/(2.*N16n);
        }
        new_data[c2x2n_id(ncols,id)] = (k2n2x2n*(old_data[c2n_id(ncols,id)] - 2*old_data[c2x2n_id(ncols,id)] + Catm[id])*N2n)/(2.*N2x2n);
        new_data[c2x4n_id(ncols,id)] = (k2x2n2x4n*(old_data[c2x2n_id(ncols,id)] - 2*old_data[c2x4n_id(ncols,id)] + Catm[id])*N2x2n)/(2.*N2x4n);
        new_data[c2x8n_id(ncols,id)] = (k2x4n2x8n*(old_data[c2x4n_id(ncols,id)] - 2*old_data[c2x8n_id(ncols,id)] + Catm[id])*N2x4n)/(2.*N2x8n);
        if (N2x16nb  <TH_NULL){
            new_data[c2x16n_id(ncols,id)] = new_data[c2x8n_id(ncols,id)];
        }else{
            new_data[c2x16n_id(ncols,id)] =(k2x8n2x16n*(-2*old_data[c2x16n_id(ncols,id)] + old_data[c2x8n_id(ncols,id)] + Catm[id])*N2x8n)/(2.*N2x16n);
        }
        return 0;
}



int  POP8_h_exp_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n  )
{
    int error = 0;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<nrows;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n;
        double dtpMono = 0;
        calc_populations_h(id, ncols, indata, t, n2_0, n4_0, n8_0 , t0_2n , p2n, dtp2n, p4n, dtp4n, p8n, dtp8n);    
        if (rhs( ncols,  id, Catm,  old_data,  new_data,  d, p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMonoS, dtpMono  )) return 1;

     }
     return 0; 
}

int  POP8_d_exp_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double p2nS, double p4nS, double pMonoSD   )
{
    int error = 0;
    double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<8;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        calc_populations_d(id, ncols, indata, t, p2nS, p4nS, n2_0 , n4_0 , n8_0, t0_2n, pMonoS, pMonoSD,
                           p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono);
        if (rhs( ncols,  id, Catm,  old_data,  new_data,  d, p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono  )) return 1;

     }
     return 0; 
}

int  POP8P_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   )
{
    int error = 0;
    double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<8;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        double p2nS = indata[p2nS_id(ncols,id)];
        double p4nS = indata[p4nS_id(ncols,id)];
        calc_populations_d(id, ncols, indata, t, p2nS, p4nS, n2_0 , n4_0 , n8_0, t0_2n, pMonoS, pMonoSD,
                           p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono);
        if (rhs( ncols,  id, Catm,  old_data,  new_data,  d, p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono  )) return 1;

     }
     return 0; 
}


int  POP8P_H2b_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double da, double db, double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   )
{
    int error = 0;
    double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono, d_time;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<8;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        if ( t > indata[age_d_id(ncols,id)]  ) d_time= db;
        else d_time = da;
        double p2nS = indata[p2nS_id(ncols,id)];
        double p4nS = indata[p4nS_id(ncols,id)];
        calc_populations_d(id, ncols, indata, t, p2nS, p4nS, n2_0 , n4_0 , n8_0, t0_2n, pMonoS, pMonoSD,
                           p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono);
        if (rhs( ncols,  id, Catm,  old_data,  new_data,  d_time, p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono  )) return 1;

     }
     return 0; 
}




int rhs_d(int ncols, int id, double* Catm, double* old_data, double* new_data,
        double d, double k4n2x4n, double k8n2x8n, double k16n2x16n,
        double b4n, double b8n, double b16n, double b2x2n, double b2x4n, double b2x8n, double b2x16n,
        double p2n, double dtp2n, double p4n, double dtp4n, double p8n, double dtp8n, double pMono, double dtpMono ){

        double N2n = Ntotal*p2n*pMono;
        double N4n = Ntotal*p4n*pMono;
        double N8n = Ntotal*p8n*pMono;
        double N16nb = -(Ntotal*(-1 + p2n + p4n + p8n)*pMono);
        double N2x2n = -(Ntotal*p2n*(-1 + pMono));
        double N2x4n = -(Ntotal*p4n*(-1 + pMono));
        double N2x8n = -(Ntotal*p8n*(-1 + pMono));
        double N2x16nb = Ntotal*(-1 + p2n + p4n + p8n)*(-1 + pMono);

        double N16n   = N16nb  <TH_NULL ? TH_NULL : N16nb;
        double N2x16n = N2x16nb<TH_NULL ? TH_NULL : N2x16nb;


        double dtN2n = Ntotal*pMono*dtp2n + p2n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN4n = Ntotal*pMono*dtp4n + p4n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN8n = Ntotal*pMono*dtp8n + p8n*(pMono*dtNtotal + Ntotal*dtpMono);
        double dtN16n = -(pMono*((-1 + p2n + p4n + p8n)*dtNtotal + Ntotal*(dtp2n + dtp4n + dtp8n))) - Ntotal*(-1 + p2n + p4n + p8n)*dtpMono;
        double dtN2x2n = -((-1 + pMono)*(p2n*dtNtotal + Ntotal*dtp2n)) - Ntotal*p2n*dtpMono;
        double dtN2x4n = -((-1 + pMono)*(p4n*dtNtotal + Ntotal*dtp4n)) - Ntotal*p4n*dtpMono;
        double dtN2x8n = -((-1 + pMono)*(p8n*dtNtotal + Ntotal*dtp8n)) - Ntotal*p8n*dtpMono;
        double dtN2x16n = (-1 + p2n + p4n + p8n)*(-1 + pMono)*dtNtotal + Ntotal*((-1 + pMono)*(dtp2n + dtp4n + dtp8n) + (-1 + p2n + p4n + p8n)*dtpMono);



        // implicit paras 

        double b2n = ((-b16n + d)*N16n - b2x16n*N2x16n - b2x2n*N2x2n - b2x4n*N2x4n - b2x8n*N2x8n - b4n*N4n - b8n*N8n + d*(N2n + N2x16n + N2x2n + N2x4n + N2x8n + N4n + N8n) + dtN16n + dtN2n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n + dtN4n + dtN8n)/N2n;
        double k2n4n = ((-b16n + d + k16n2x16n)*N16n + (-b4n + d + k4n2x4n)*N4n + (-b8n + d + k8n2x8n)*N8n + dtN16n + dtN4n + dtN8n)/N2n;
        double k4n8n = ((-b16n + d + k16n2x16n)*N16n + (-b8n + d + k8n2x8n)*N8n + dtN16n + dtN8n)/N4n;
        double k8n16n = ((-b16n + d + k16n2x16n)*N16n + dtN16n)/N8n;
        double k2n2x2n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n - b2x2n*N2x2n - b2x4n*N2x4n - b2x8n*N2x8n + d*(N2x2n + N2x4n + N2x8n) - k4n2x4n*N4n - k8n2x8n*N8n + dtN2x16n + dtN2x2n + dtN2x4n + dtN2x8n)/N2n;
        double k2x2n2x4n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + (-b2x4n + d)*N2x4n + (-b2x8n + d)*N2x8n - k4n2x4n*N4n - k8n2x8n*N8n + dtN2x16n + dtN2x4n + dtN2x8n)/N2x2n;
        double k2x4n2x8n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + (-b2x8n + d)*N2x8n - k8n2x8n*N8n + dtN2x16n + dtN2x8n)/N2x4n;
        double k2x8n2x16n = (-(k16n2x16n*N16n) + (-b2x16n + d)*N2x16n + dtN2x16n)/N2x8n;
        
        if ( (b2n <0) || (k2n4n <0) || (k4n8n <0) || (k8n16n <0) || (k2n2x2n <0) || (k2x2n2x4n <0) || (k2x4n2x8n<0) || (k2x8n2x16n<0) ) {
            return 1;
        }     
        if ( (N2n <0) || (N4n <0) || (N8n <0) || (N2x2n <0) || (N2x4n <0) || (N2x8n <0) || (N16nb<-TH_NULL) || (N2x16nb<-TH_NULL) ) {
            return 1;
        }
        new_data[c2n_id(ncols,id)] =b2n*(-old_data[c2n_id(ncols,id)] + Catm[id]);
        new_data[c4n_id(ncols,id)] =b4n*(-old_data[c4n_id(ncols,id)] + Catm[id]) + (k2n4n*(old_data[c2n_id(ncols,id)] - 2*old_data[c4n_id(ncols,id)] + Catm[id])*N2n)/(2.*N4n);
        new_data[c8n_id(ncols,id)] =b8n*(-old_data[c8n_id(ncols,id)] + Catm[id]) + (k4n8n*(old_data[c4n_id(ncols,id)] - 2*old_data[c8n_id(ncols,id)] + Catm[id])*N4n)/(2.*N8n);
        if (N16nb  <TH_NULL){
            new_data[c16n_id(ncols,id)] = new_data[c8n_id(ncols,id)];
        }else{
            new_data[c16n_id(ncols,id)] =(2*b16n*(-old_data[c16n_id(ncols,id)] + Catm[id])*N16n + k8n16n*(-2*old_data[c16n_id(ncols,id)] + old_data[c8n_id(ncols,id)] + Catm[id])*N8n)/(2.*N16n);
        }
        
        new_data[c2x2n_id(ncols,id)] =b2x2n*(-old_data[c2x2n_id(ncols,id)] + Catm[id]) + (k2n2x2n*(old_data[c2n_id(ncols,id)] - 2*old_data[c2x2n_id(ncols,id)] + Catm[id])*N2n)/(2.*N2x2n);
        new_data[c2x4n_id(ncols,id)] =(k2x2n2x4n*old_data[c2x2n_id(ncols,id)]*N2x2n + Catm[id]*(k2x2n2x4n*N2x2n + 2*b2x4n*N2x4n) + k4n2x4n*(old_data[c4n_id(ncols,id)] + Catm[id])*N4n - 2*old_data[c2x4n_id(ncols,id)]*(k2x2n2x4n*N2x2n + b2x4n*N2x4n + k4n2x4n*N4n))/(2.*N2x4n);
        new_data[c2x8n_id(ncols,id)] =(k2x4n2x8n*old_data[c2x4n_id(ncols,id)]*N2x4n + Catm[id]*(k2x4n2x8n*N2x4n + 2*b2x8n*N2x8n) + k8n2x8n*(old_data[c8n_id(ncols,id)] + Catm[id])*N8n - 2*old_data[c2x8n_id(ncols,id)]*(k2x4n2x8n*N2x4n + b2x8n*N2x8n + k8n2x8n*N8n))/(2.*N2x8n);
        if (N2x16nb  <TH_NULL){
            new_data[c2x16n_id(ncols,id)] = new_data[c2x8n_id(ncols,id)];
        }else{
            new_data[c2x16n_id(ncols,id)] =(k16n2x16n*old_data[c16n_id(ncols,id)]*N16n + Catm[id]*(k16n2x16n*N16n + 2*b2x16n*N2x16n) + k2x8n2x16n*(old_data[c2x8n_id(ncols,id)] + Catm[id])*N2x8n - 2*old_data[c2x16n_id(ncols,id)]*(k16n2x16n*N16n + b2x16n*N2x16n + k2x8n2x16n*N2x8n))/(2.*N2x16n);
        }
        return 0;
}

int  POP8P_D_h_exp_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double k4n2x4n, double k8n2x8n, double k16n2x16n,
    double b4n, double b8n, double b16n, double b2x2n, double b2x4n, double b2x8n, double b2x16n,
    double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n )
{
        int error = 0;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<nrows;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n =0;
        double dtpMono = 0;
        calc_populations_h(id, ncols, indata, t, n2_0, n4_0, n8_0 , t0_2n , p2n, dtp2n, p4n, dtp4n, p8n, dtp8n);
        if (rhs_d( ncols,  id, Catm,  old_data,  new_data,  d, 
        k4n2x4n, k8n2x8n, k16n2x16n, b4n, b8n, b16n, b2x2n, b2x4n, b2x8n, b2x16n,
        p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMonoS, dtpMono  )) return 1;

     }
     return 0; 
          
}

int  POP8P_D_d_exp_Hcalc_rhs(int nrows, int ncols, double* indata,int nc, double* Catm, int no, double* old_data, int nn, double* new_data, 
    double t, double d, double k4n2x4n, double k8n2x8n, double k16n2x16n,
    double b4n, double b8n, double b16n, double b2x2n, double b2x4n, double b2x8n, double b2x16n,
    double pMonoS,double n2_0 ,double n4_0 ,double n8_0 ,double t0_2n, double pMonoSD   )
{
    int error = 0;
    double p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono;
     for (unsigned int id;id<ncols;id++){
        if ( t > (indata[age_id(ncols,id)] + EPS) ){
            for (unsigned int j=0;j<8;j++){
                new_data[j*ncols + id] = 0;
            }
            continue;
        }
        double p2nS = indata[p2nS_id(ncols,id)];
        double p4nS = indata[p4nS_id(ncols,id)];
        calc_populations_d(id, ncols, indata, t, p2nS, p4nS, n2_0 , n4_0 , n8_0, t0_2n, pMonoS, pMonoSD,
                           p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono);
        if (rhs_d( ncols,  id, Catm,  old_data,  new_data,  d, 
        k4n2x4n, k8n2x8n, k16n2x16n, b4n, b8n, b16n, b2x2n, b2x4n, b2x8n, b2x16n,
        p2n, dtp2n, p4n, dtp4n, p8n, dtp8n, pMono, dtpMono  )) return 1;

     }
     return 0; 
}


