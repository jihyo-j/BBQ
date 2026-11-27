/* refio -- replay the refinement selection rules against real SSD reads.
 *
 * Every arm reads the SAME pool, in the same dhat order, from the same
 * page-aligned O_DIRECT file; the arms differ in (a) which pool members they
 * read and (b) whether the reads are issued one at a time or as one batch.
 *
 *   seq_bound   eps : RaBitQ's rule. theta = k-th best exact distance so far,
 *                     read x iff dhat(x)-eps*rad(x) < theta.  Each decision
 *                     needs the previous read's result -> QD 1, dependent.
 *   seq_calib   tau : calibrated prefix {dhat <= dhat_k + tau}, read one at a
 *                     time through the same path (isolates the rule alone).
 *   batch_calib tau : the same prefix, submitted as one io_uring batch.
 *   batch_boundM eps: {dhat - dhat_k < eps*rad}, one io_uring batch.
 *   twophase  eps   : batch of the first k, then one batch under exact theta.
 *   batch_fixedR R  : the top-R prefix, one io_uring batch.
 *   chunk_bound eps : the same bound, but the pool is walked in chunks of QD
 *                     candidates; the reads a chunk selects go out as one
 *                     batch and theta is updated once per chunk.  QD=1 is
 *                     seq_bound, QD>=ef is the bound with no feedback at all.
 *                     This is the "RaBitQ could batch too" arm.
 *
 * Build: gcc -O2 -march=native refio.c -luring -lm -o refio
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <liburing.h>

static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec+1e-9*t.tv_nsec;}

typedef struct{int nq,ef,D,K;float*Q;int*ids;float*dhat;float*rad;}Pool;

static Pool load_pool(const char*p){
  FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}
  Pool P;int h[4];
  if(fread(h,4,4,f)!=4){fprintf(stderr,"hdr\n");exit(1);}
  P.nq=h[0];P.ef=h[1];P.D=h[2];P.K=h[3];
  size_t nq=P.nq,ef=P.ef,D=P.D;
  P.Q=malloc(nq*D*4);P.ids=malloc(nq*ef*4);P.dhat=malloc(nq*ef*4);P.rad=malloc(nq*ef*4);
  if(fread(P.Q,4,nq*D,f)!=nq*D||fread(P.ids,4,nq*ef,f)!=nq*ef||
     fread(P.dhat,4,nq*ef,f)!=nq*ef||fread(P.rad,4,nq*ef,f)!=nq*ef){
    fprintf(stderr,"short pool file\n");exit(1);}
  fclose(f);return P;
}

/* k-best max-at-top array, k<=64 */
typedef struct{double v[64];int n,k;}Top;
static void top_init(Top*t,int k){t->n=0;t->k=k;}
static double top_theta(Top*t){return t->n<t->k?INFINITY:t->v[0];}
static void top_push(Top*t,double d){
  if(t->n<t->k){ /* insert, keep max at 0 by simple scan */
    t->v[t->n++]=d;
    if(t->n==t->k){ /* move max to slot 0 */
      int m=0;for(int i=1;i<t->n;i++)if(t->v[i]>t->v[m])m=i;
      double tmp=t->v[0];t->v[0]=t->v[m];t->v[m]=tmp;}
    return;}
  if(d<t->v[0]){t->v[0]=d;int m=0;
    for(int i=1;i<t->n;i++)if(t->v[i]>t->v[m])m=i;
    double tmp=t->v[0];t->v[0]=t->v[m];t->v[m]=tmp;}
}

static double l2sq(const float*a,const float*b,int D){
  double s=0;for(int i=0;i<D;i++){double d=(double)a[i]-b[i];s+=d*d;}return s;}

int main(int argc,char**argv){
  if(argc<7){fprintf(stderr,
    "usage: refio POOLFILE VECFILE PAGESZ ARM PARAM NQ [QD]\n");return 1;}
  const char*pf=argv[1],*vf=argv[2];
  size_t pagesz=atol(argv[3]);
  const char*arm=argv[4];double param=atof(argv[5]);
  int nq_use=atoi(argv[6]);int qd=argc>7?atoi(argv[7]):0;
  Pool P=load_pool(pf);
  if(nq_use<=0||nq_use>P.nq)nq_use=P.nq;
  int fd=open(vf,O_RDONLY|O_DIRECT);
  if(fd<0){perror(vf);return 1;}
  int ef=P.ef,D=P.D,K=P.K;
  size_t pages=pagesz/4096;
  void*buf;if(posix_memalign(&buf,4096,pagesz*(size_t)ef)){perror("memalign");return 1;}
  struct io_uring ring;
  int want_ring=strncmp(arm,"batch",5)==0||strncmp(arm,"twophase",8)==0
                ||strncmp(arm,"chunk",5)==0;
  if(want_ring&&io_uring_queue_init(4096,&ring,0)<0){perror("uring");return 1;}
  int*sel=malloc(ef*sizeof(int));
  double*lat=malloc(nq_use*sizeof(double));
  long long tot_reads=0,tot_pages=0;int tot_rounds=0;
  double t_all=now();
  for(int q=0;q<nq_use;q++){
    const float*qv=P.Q+(size_t)q*D;
    const int*ids=P.ids+(size_t)q*ef;
    const float*dh=P.dhat+(size_t)q*ef;
    const float*rd=P.rad+(size_t)q*ef;
    double dk=dh[K-1];
    double t0=now();long reads=0;int rounds=0;
    Top T;top_init(&T,K);
    if(strcmp(arm,"seq_bound")==0||strcmp(arm,"seq_calib")==0){
      int isb=strcmp(arm,"seq_bound")==0;
      for(int j=0;j<ef;j++){
        if(ids[j]<0)continue;
        int take=isb?((dh[j]-param*rd[j])<top_theta(&T)):((dh[j]-dk)<=param);
        if(!take)continue;
        if(pread(fd,buf,pagesz,(off_t)ids[j]*pagesz)!=(ssize_t)pagesz){
          perror("pread");return 1;}
        reads++;rounds++;
        top_push(&T,l2sq(qv,(const float*)buf,D));
      }
    }else if(strcmp(arm,"chunk_bound")==0){
      int C=qd>0?qd:1;
      for(int base=0;base<ef;base+=C){
        int hi=base+C<ef?base+C:ef,n=0;
        double th=top_theta(&T);
        for(int j=base;j<hi;j++)
          if(ids[j]>=0&&(dh[j]-param*rd[j])<th)sel[n++]=j;
        if(!n)continue;
        for(int i=0;i<n;i++){
          struct io_uring_sqe*s=io_uring_get_sqe(&ring);
          io_uring_prep_read(s,fd,(char*)buf+pagesz*(size_t)i,pagesz,
                             (off_t)ids[sel[i]]*pagesz);
          io_uring_sqe_set_data64(s,i);
        }
        io_uring_submit(&ring);
        for(int i=0;i<n;i++){struct io_uring_cqe*c;io_uring_wait_cqe(&ring,&c);
          io_uring_cqe_seen(&ring,c);}
        rounds++;
        for(int i=0;i<n;i++)
          top_push(&T,l2sq(qv,(const float*)((char*)buf+pagesz*(size_t)i),D));
        reads+=n;
      }
    }else{
      /* ---- batched arms: build the selection, then one (or two) io_uring batches */
      int n=0;
      if(strcmp(arm,"batch_calib")==0){
        for(int j=0;j<ef;j++)if(ids[j]>=0&&(dh[j]-dk)<=param)sel[n++]=j;
      }else if(strcmp(arm,"batch_boundM")==0){
        for(int j=0;j<ef;j++)if(ids[j]>=0&&(dh[j]-dk)<param*rd[j])sel[n++]=j;
      }else if(strcmp(arm,"batch_fixedR")==0){
        int R=(int)param;for(int j=0;j<ef&&j<R;j++)if(ids[j]>=0)sel[n++]=j;
      }else if(strcmp(arm,"twophase")==0){
        for(int j=0;j<K&&j<ef;j++)if(ids[j]>=0)sel[n++]=j;
      }else{fprintf(stderr,"unknown arm %s\n",arm);return 1;}
      int chunk=qd>0?qd:n;
      for(int off=0;off<n;off+=chunk){
        int m=n-off<chunk?n-off:chunk;
        for(int i=0;i<m;i++){
          struct io_uring_sqe*s=io_uring_get_sqe(&ring);
          io_uring_prep_read(s,fd,(char*)buf+pagesz*(size_t)(off+i),pagesz,
                             (off_t)ids[sel[off+i]]*pagesz);
          io_uring_sqe_set_data64(s,off+i);
        }
        io_uring_submit(&ring);
        for(int i=0;i<m;i++){
          struct io_uring_cqe*c;int r=io_uring_wait_cqe(&ring,&c);
          if(r<0||c->res!=(int)pagesz){fprintf(stderr,"cqe %d %d\n",r,c?c->res:0);return 1;}
          io_uring_cqe_seen(&ring,c);
        }
        rounds++;
      }
      for(int i=0;i<n;i++)top_push(&T,l2sq(qv,(const float*)((char*)buf+pagesz*(size_t)i),D));
      reads=n;
      if(strcmp(arm,"twophase")==0){
        double th=top_theta(&T);int n2=0;
        for(int j=K;j<ef;j++)if(ids[j]>=0&&(dh[j]-param*rd[j])<th)sel[n2++]=j;
        for(int i=0;i<n2;i++){
          struct io_uring_sqe*s=io_uring_get_sqe(&ring);
          io_uring_prep_read(s,fd,(char*)buf+pagesz*(size_t)i,pagesz,
                             (off_t)ids[sel[i]]*pagesz);
          io_uring_sqe_set_data64(s,i);
        }
        if(n2){io_uring_submit(&ring);
          for(int i=0;i<n2;i++){struct io_uring_cqe*c;io_uring_wait_cqe(&ring,&c);
            io_uring_cqe_seen(&ring,c);}
          rounds++;
          for(int i=0;i<n2;i++)top_push(&T,l2sq(qv,(const float*)((char*)buf+pagesz*(size_t)i),D));
          reads+=n2;}
      }
    }
    lat[q]=now()-t0;tot_reads+=reads;tot_pages+=reads*(long long)pages;tot_rounds+=rounds;
  }
  double wall=now()-t_all;
  /* p50/p95/p99 */
  for(int i=1;i<nq_use;i++){double v=lat[i];int j=i-1;
    while(j>=0&&lat[j]>v){lat[j+1]=lat[j];j--;}lat[j+1]=v;}
  printf("{\"arm\":\"%s\",\"param\":%g,\"nq\":%d,\"qd\":%d,"
         "\"reads\":%.4f,\"pages\":%.4f,\"rounds\":%.4f,"
         "\"lat_mean_us\":%.2f,\"lat_p50_us\":%.2f,\"lat_p95_us\":%.2f,"
         "\"lat_p99_us\":%.2f,\"qps_refine\":%.2f,\"wall_s\":%.3f}\n",
         arm,param,nq_use,qd,(double)tot_reads/nq_use,(double)tot_pages/nq_use,
         (double)tot_rounds/nq_use,1e6*wall/nq_use,1e6*lat[nq_use/2],
         1e6*lat[(int)(0.95*nq_use)],1e6*lat[(int)(0.99*nq_use)],
         nq_use/wall,wall);
  return 0;
}
