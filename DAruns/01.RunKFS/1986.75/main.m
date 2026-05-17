function [obs,sim]=main(strdate)
global Lind Iind Jind;
if nargin==0
    strdate='1986.75';
end
areg=0.11;
method=1;

posfolds=ls(['..\' strdate '\*hydrol*dep*']);
%pfile=['..\' strdate '\' posfold '\lastconc.xls'];

 obsready=false;

if obsready
    s=load('obs2pos.mat');
    obs=s.obs;brefnum=s.brefnum;breflev=s.breflev; boxlats=s.boxlats;boxlons=s.boxlons;
else
    [obs,brefnum,breflev,box_vs_allobs,boxlats,boxlons]=obs_vs_poseidon([strdate '.dat'],'../../../measurements.exp2','Box_surfareas.xlsx');
    save('obs2pos.mat','obs','brefnum','breflev','boxlats','boxlons');
    xlswrite(['out_' strdate  '.xlsx'],cat(2,obs,brefnum),'obs');
end
Nobs=size(obs,1);%inverse obs error covariance matr (it's square root)
invO2=zeros(Nobs,Nobs)
for i=1:Nobs
    invO2(i,i)=1/areg/obs(i);
end


[bnlev]=box_levs();
[BoxNeib]=Box_Neighbors(size(bnlev,2),brefnum,boxlats,boxlons);
[Lind,Iind,Jind]=IJ_Lind(bnlev)
Np=size(Iind,2);

[Hmatr]=init_Hmatr(brefnum,breflev,Lind);

Nens=size(posfolds,1);
Ens=zeros(Np,Nens);
for i=1:Nens;;
    pfile=['..\' strdate '\' posfolds(i,:) '\lastconc.xls'];
    [arr,pheader,pfooter,sed]=read_poseidon(pfile,bnlev);
    for L=1:Np
        Ens(L,i)=arr(Iind(L),Jind(L));
        Sens{i}=sed;
    end
end
CC=cov(Ens');
[CC1,LocM]=regularize_Covmatr(CC,Iind,Jind,[1 2 3 49 50]);%LocM is localization matrix
% additional localization (regularization): 
if true
for L=1:Np
    for K=L+1:Np
        boxL=Iind(L);
        levL=Jind(L);
        
        boxK=Iind(K);
        levK=Jind(K);
        
        if CC1(L,K)==0
            LocM(L,K)=0;LocM(K,L)=0;
            continue;
        end
        
        if BoxNeib(boxL,boxK)==0
            LocM(L,K)=0;
            LocM(K,L)=0;
            CC1(L,K)=0;
            CC1(K,L)=0;
            continue;
        end
        if abs(levK-levL)>1
            LocM(L,K)=0;
            LocM(K,L)=0;            
            CC1(L,K)=0;
            CC1(K,L)=0;
            continue;
        end
        if boxL==boxK | levK==levL % nonzero covariance either for neighbor boxes at same level or for neighbor levels at same box
            LocM(L,K)=0.2;
            LocM(K,L)=0.2;
            CC1(L,K)=0.2*CC1(L,K);
             CC1(K,L)=0.2*CC1(L,K);            
             continue;
        else
            LocM(L,K)=0;
            LocM(K,L)=0;            
            CC1(L,K)=0;
            CC1(K,L)=0;
            continue;            
        end
        continue
    end
end
end
Covr_1=inv(CC1); %inverse background covariance matrix

[Lmatr,Dmatr] = ldl(Covr_1);

Dsqrt=Dmatr.^0.5;

Gmatr_p1=invO2*Hmatr;
Gmatr_p2=Dsqrt*(Lmatr');
     
Gmodif=cat(1,Gmatr_p1,Gmatr_p2);
 
%create perturbed observations
obs_noise=zeros(Nobs,Nens);
for ii=1:Nobs
    for jj=1:Nens
        obs_noise(ii,jj)=obs(ii)+(1/invO2(ii,ii))*randn;
    end
end
%ymodif1=invO2*obs;

for i=1:Nens
    ymodif1=invO2*obs_noise(:,i);

    ymodif2=Dsqrt*(Lmatr')*Ens(:,i);

    ymodif=cat(1,ymodif1,ymodif2);

    if method==1
         sol(:,i) = lsqnonneg(Gmodif,ymodif);
      elseif method==2
           lb(1:Nx,1)=1;% very small value to avoid less than zero sols
           sumvect(1:Nmeas)=1;
           Aeq=sumvect*Hmatr;
            beq=sum(obs);            
            sol(:,i)=lsqlin(Gmodif,ymodif,[],[],Aeq,beq,lb,[]);
        else
             'Method is unknown' 
              method
              pause;
    end
    
    mkdir(posfolds(i,:));
    ofname=[posfolds(i,:) '\Initconc.dat'];
    write_poseidon(ofname, sol(:,i) ,Lind,bnlev,pheader, pfooter,Sens{i});
    norm(sol);
    
    xlswrite(['out_' strdate  '.xlsx'],Hmatr*sol(:,i),posfolds(i,:));
                         %fprintf(fid,'%d %f\n',iter,norm(sol));
     
                         %plot(xn',Qb,xn',sol);

end

ens=mean(sol,2);
xlswrite(['out_' strdate  '.xlsx'],Hmatr*ens,'Ens');
    

                                

 

  

end
