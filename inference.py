# -*- coding: utf-8 -*-
"""
Created on Wed May 04 22:40:52 2017

@author: Victor Suárez Gutiérrez
Research Assistant. Data Scientist at URJC/HGUGM.
Contact: ssuarezvictor@gmail.com
"""


###############################################################################
###############################################################################
###############################################################################
###############################################################################
                            # San Silvestre race:
###############################################################################
###############################################################################
###############################################################################
###############################################################################

# Inicialization:
#%reset -f


# Define working directory
import os
os.chdir('C:/Users/Victor/Documents/Ambito_profesional/proyectos/s_s_race')


# import libraries
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sst
  


###############################################################################
###############################################################################
# Load Dataset:  
data = pd.read_csv('resultados_popular.csv',sep=';')
total_time2 = pd.to_datetime(list(map((lambda x: datetime.strptime(x, '%H:%M:%S')), data['Tiempo'])), format='%H/%M/%S')
data['Tiempo'] = total_time2.hour*3600+total_time2.minute*60+total_time2.second
del data['Ritmo']



###############################################################################
###############################################################################
# Data preprocessing:
categories = pd.Series(data['Categoria'].unique())
nans = data.ix[data['Categoria'].isnull(),:]
data.ix[nans.index,'Categoria'] = 'Unknown'
data.ix[nans.index,'PuestoCategoria'] = np.arange(1,nans.shape[0]+1)
categories = pd.Series(data['Categoria'].unique())

# Question: We want to manage another race similar to San Silvestre race taking shorter time.

###############################################################################
###############################################################################
# Boxplot all categories:
data.boxplot(column='Tiempo', by='Categoria', ax=None, fontsize=None, rot=0, grid=True, figsize=None, layout=None, return_type=None)
   

limit = 5000  # seconds. APROX: MEAN+2*STD.
###############################################################################     
# JunF Inference
JunF = np.log(data.ix[data.ix[:,'Categoria']=='JunF','Tiempo'])
normal = pd.DataFrame(sst.norm(np.mean(JunF), np.std(JunF)).rvs(len(JunF)))
normal.plot(kind='density', color='red')
JunF.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(JunF.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('JunF PDF')
plt.legend(['Normal PDF', 'JunF PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(JunF)
sst.kstest(JunF, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_JunF = sst.norm(np.mean(JunF), np.std(JunF)).cdf(np.log(limit))
outliers_JunF = int(sum(JunF>np.mean(JunF)+3*np.std(JunF)))   #99% confidence= +-3std over mean.      
out_JunF = int(np.round(JunF.shape[0]*(1-px_JunF)-outliers_JunF))
p_JunF = JunF.shape[0]/data.shape[0]

###############################################################################
# SenF Inference
SenF = data.ix[data.ix[:,'Categoria']=='SenF','Tiempo']
normal = pd.DataFrame(sst.norm(np.mean(SenF), np.std(SenF)).rvs(len(SenF)))
normal.plot(kind='density', color='red')
SenF.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(SenF.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('SenF PDF')
plt.legend(['Normal PDF', 'SenF PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
dist = sst.norm
args = dist.fit(SenF)
sst.kstest(SenF, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_SenF = sst.norm(np.mean(SenF), np.std(SenF)).cdf(limit)
outliers_SenF = sum(SenF>np.mean(SenF)+3*np.std(SenF))   #99% confidence= +-3std over mean.
out_SenF = int(np.round(SenF.shape[0]*(1-px_SenF)-outliers_SenF))
p_SenF = SenF.shape[0]/data.shape[0]

###############################################################################
# VtM45 Inference
VtM45 = np.log(data.ix[data.ix[:,'Categoria']=='VtM45','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(VtM45), np.std(VtM45)).rvs(len(VtM45)))
normal.plot(kind='density', color='red')
VtM45.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtM45.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtM45 PDF')
plt.legend(['Normal PDF', 'VtM45 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)
    
dist = sst.norm
args = dist.fit(VtM45)
sst.kstest(VtM45, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_VtM45 = sst.norm(np.mean(VtM45), np.std(VtM45)).cdf(np.log(limit))
outliers_VtM45 = sum(VtM45>np.mean(VtM45)+3*np.std(VtM45))   #99% confidence= +-3std over mean.                     
out_VtM45 = int(np.round(VtM45.shape[0]*(1-px_VtM45)-outliers_VtM45))
p_VtM45 = VtM45.shape[0]/data.shape[0]

###############################################################################
# VtF45 Inference
VtF45 = np.log(data.ix[data.ix[:,'Categoria']=='VtF45','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(VtF45), np.std(VtF45)).rvs(len(VtF45)))
normal.plot(kind='density', color='red')
VtF45.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtF45.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtF45 PDF')
plt.legend(['Normal PDF', 'VtF45 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(VtF45)
sst.kstest(VtF45, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_VtF45 = sst.norm(np.mean(VtF45), np.std(VtF45)).cdf(np.log(limit))
outliers_VtF45 = sum(VtF45>np.mean(VtF45)+3*np.std(VtF45))   #99% confidence= +-3std over mean.                      
out_VtF45 = int(np.round(VtF45.shape[0]*(1-px_VtF45)-outliers_VtF45))
p_VtF45 = VtF45.shape[0]/data.shape[0]

###############################################################################     
# VtF55 Inference
VtF55 = np.log(data.ix[data.ix[:,'Categoria']=='VtF55','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(VtF55), np.std(VtF55)).rvs(len(VtF55)))
normal.plot(kind='density', color='red')
VtF55.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtF55.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtF55 PDF')
plt.legend(['Normal PDF', 'VtF55 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(VtF55)
sst.kstest(VtF55, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_VtF55 = sst.norm(np.mean(VtF55), np.std(VtF55)).cdf(np.log(limit))
outliers_VtF55 = sum(VtF55>np.mean(VtF55)+3*np.std(VtF55))   #99% confidence= +-3std over mean.         
out_VtF55 = int(np.round(VtF55.shape[0]*(1-px_VtF55)-outliers_VtF55))
p_VtF55 = VtF55.shape[0]/data.shape[0]

###############################################################################     
# VtM55 Inference
VtM55 = np.log(data.ix[data.ix[:,'Categoria']=='VtM55','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(VtM55), np.std(VtM55)).rvs(len(VtM55)))
normal.plot(kind='density', color='red')
VtM55.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtM55.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtM55 PDF')
plt.legend(['Normal PDF', 'VtM55 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(VtM55)
sst.kstest(VtM55, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_VtM55 = sst.norm(np.mean(VtM55), np.std(VtM55)).cdf(np.log(limit))
outliers_VtM55 = sum(VtM55>np.mean(VtM55)+3*np.std(VtM55))   #99% confidence= +-3std over mean.                              
out_VtM55 = int(np.round(VtM55.shape[0]*(1-px_VtM55)-outliers_VtM55))
p_VtM55 = VtM55.shape[0]/data.shape[0]

###############################################################################     
# SenM Inference
SenM = data.ix[data.ix[:,'Categoria']=='SenM','Tiempo']   # log over dataset.
dist = sst.gamma
args = dist.fit(SenM)
dgamma = pd.DataFrame(sst.gamma(args[0], args[1], args[2]).rvs(len(SenM)))
dgamma.plot(kind='density', color='red')
SenM.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(SenM.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('SenM PDF')
plt.legend(['Normal PDF', 'SenM PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

sst.kstest(SenM, 'gamma', args)      # alpha=0,01.      H0 is accepted. 
px_SenM = sst.gamma(args[0], args[1], args[2]).cdf(limit)
outliers_SenM = sum(SenM>np.mean(SenM)+3*np.std(SenM))   #99% confidence= +-3std over mean.                     
out_SenM = int(np.round(SenM.shape[0]*(1-px_SenM)-outliers_SenM))
p_SenM = SenM.shape[0]/data.shape[0]
                     
###############################################################################     
# VtM35 Inference
VtM35 = data.ix[data.ix[:,'Categoria']=='VtM35','Tiempo']   # log over dataset.
dist = sst.gamma
args = dist.fit(VtM35)
dgamma = pd.DataFrame(sst.gamma(args[0], args[1], args[2]).rvs(len(VtM35)))
dgamma.plot(kind='density', color='red')
VtM35.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtM35.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtM35 PDF')
plt.legend(['Normal PDF', 'VtM35 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

sst.kstest(VtM35, 'gamma', args)      # alpha=0,01.      H0 is accepted. 
px_VtM35 = sst.norm(np.mean(VtM35), np.std(VtM35)).cdf(limit)
outliers_VtM35 = sum(VtM35>np.mean(VtM35)+3*np.std(VtM35))   #99% confidence= +-3std over mean.               
out_VtM35 = int(np.round(VtM35.shape[0]*(1-px_VtM35)-outliers_VtM35))
p_VtM35 = VtM35.shape[0]/data.shape[0]

###############################################################################     
# VtF35 Inference
VtF35 = data.ix[data.ix[:,'Categoria']=='VtF35','Tiempo']   # log over dataset.
dist = sst.gamma
args = dist.fit(VtF35)
dgamma = pd.DataFrame(sst.gamma(args[0], args[1], args[2]).rvs(len(VtF35)))
dgamma.plot(kind='density', color='red')
VtF35.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(VtF35.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('VtF35 PDF')
plt.legend(['Normal PDF', 'VtF35 PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

sst.kstest(VtF35, 'gamma', args)      # alpha=0,01.      H0 is accepted. 
px_VtF35 = sst.norm(np.mean(VtF35), np.std(VtF35)).cdf(limit)
outliers_VtF35 = sum(VtF35>np.mean(VtF35)+3*np.std(VtF35))   #99% confidence= +-3std over mean.
out_VtF35 = int(np.round(VtF35.shape[0]*(1-px_VtF35)-outliers_VtF35))
p_VtF35 = VtF35.shape[0]/data.shape[0]

###############################################################################     
# JunM Inference
JunM = np.log(data.ix[data.ix[:,'Categoria']=='JunM','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(JunM), np.std(JunM)).rvs(len(JunM)))
normal.plot(kind='density', color='red')
JunM.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(JunM.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('JunM PDF')
plt.legend(['Normal PDF', 'JunM PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(JunM)
sst.kstest(JunM, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_JunM = sst.norm(np.mean(JunM), np.std(JunM)).cdf(np.log(limit))
outliers_JunM = sum(JunM>np.mean(JunM)+3*np.std(JunM))   #99% confidence= +-3std over mean.   
out_JunM = int(np.round(JunM.shape[0]*(1-px_JunM)-outliers_JunM))
p_JunM = JunM.shape[0]/data.shape[0]

###############################################################################     
# PromM Inference
PromM = np.log(data.ix[data.ix[:,'Categoria']=='PromM','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(PromM), np.std(PromM)).rvs(len(PromM)))
normal.plot(kind='density', color='red')
PromM.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(PromM.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('PromM PDF')
plt.legend(['Normal PDF', 'PromM PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(PromM)
sst.kstest(PromM, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_PromM = sst.norm(np.mean(PromM), np.std(PromM)).cdf(np.log(limit))
outliers_PromM = sum(PromM>np.mean(PromM)+3*np.std(PromM))   #99% confidence= +-3std over mean.   
out_PromM = int(np.round(PromM.shape[0]*(1-px_PromM)-outliers_PromM))
p_PromM = PromM.shape[0]/data.shape[0]

###############################################################################     
# PromF Inference
PromF = np.log(data.ix[data.ix[:,'Categoria']=='PromF','Tiempo'])   # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(PromF), np.std(PromF)).rvs(len(PromF)))
normal.plot(kind='density', color='red')
PromF.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(PromF.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('PromF PDF')
plt.legend(['Normal PDF', 'PromF PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(PromF)
sst.kstest(PromF, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_PromF = sst.norm(np.mean(PromF), np.std(PromF)).cdf(np.log(limit))
outliers_PromF = sum(PromF>np.mean(PromF)+3*np.std(PromF))   #99% confidence= +-3std over mean.   
out_PromF = int(np.round(PromF.shape[0]*(1-px_PromF)-outliers_PromF))
p_PromF = PromF.shape[0]/data.shape[0]

###############################################################################     
# Unknown Inference
Unknown = data.ix[data.ix[:,'Categoria']=='Unknown','Tiempo']  # log over dataset.
normal = pd.DataFrame(sst.norm(np.mean(Unknown), np.std(Unknown)).rvs(len(Unknown)))
normal.plot(kind='density', color='red')
Unknown.plot(kind='hist', bins=int(1.5*np.round(np.sqrt(Unknown.shape[0]))), normed=True, color='gray')
plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
plt.title('Unknown PDF')
plt.legend(['Normal PDF', 'Unknown PDF'])
for spine in plt.gca().spines.values():
    spine.set_visible(False)

dist = sst.norm
args = dist.fit(Unknown)
sst.kstest(Unknown, 'norm', args)      # alpha=0,01.      H0 is accepted. 
px_Unknown = sst.norm(np.mean(Unknown), np.std(Unknown)).cdf(limit)
outliers_Unknown = sum(Unknown>np.mean(Unknown)+3*np.std(Unknown))   #99% confidence= +-3std over mean.   
out_Unknown = int(np.round(Unknown.shape[0]*(1-px_Unknown)-outliers_Unknown))
p_Unknown = Unknown.shape[0]/data.shape[0]
               
    
###############################################################################     
###############################################################################        
out_PromM_total = outliers_PromM+out_PromM
out_PromF_total = outliers_PromF+out_PromF
out_SenM_total = outliers_SenM+out_SenM
out_SenF_total = outliers_SenF+out_SenF
out_VtM35_total = outliers_VtM35+out_VtM35
out_VtF35_total = outliers_VtF35+out_VtF35
out_VtM45_total = outliers_VtM45+out_VtM45
out_VtF45_total = outliers_VtF45+out_VtF45
out_VtM55_total = outliers_VtM55+out_VtM55
out_VtF55_total = outliers_VtF55+out_VtF55
out_JunM_total = outliers_JunM+out_JunM
out_JunF_total = outliers_JunF+out_JunF
out_Unknown_total = outliers_Unknown+out_Unknown
n_out_total = np.sum([out_PromM_total, out_PromF_total, out_SenM_total, out_SenF_total, out_VtM35_total, out_VtF35_total,
                  out_VtM45_total, out_VtF45_total, out_VtM55_total, out_VtF55_total, out_JunM_total, out_JunF_total,
                  out_Unknown_total])
plt.figure()
plt.scatter(np.arange(13), [out_PromM_total,out_PromF_total,out_SenM_total,out_SenF_total,
            out_VtM35_total,out_VtF35_total,out_VtM45_total,out_VtF45_total,out_VtM55_total,
            out_VtF55_total,out_JunM_total,out_JunF_total,out_Unknown_total], c=np.arange(13))
plt.title('Number of run out of %i seconds' % limit)
plt.xticks(range(13), ['PromM', 'PromF', 'SenM', 'SenF', 'VtM35', 'VtF35', 'VtM45', 'VtF45', 'VtM55', 'VtF55', 'JunM', 'JunF', 'Unknown'], color='black', rotation=30)
for spine in plt.gca().spines.values():
    spine.set_visible(False)      
    
# Reduce probability of run out dropping categories:
p_out_limit = n_out_total/data.shape[0]
p_out_limit_updated = (n_out_total-out_SenF_total)/(data.shape[0]-SenF.shape[0])
p_out_limit_updated2 = (n_out_total-out_SenF_total-out_VtF35_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0])
p_out_limit_updated3 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0])
p_out_limit_updated4 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0])
p_out_limit_updated5 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0])
p_out_limit_updated6 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0])
p_out_limit_updated7 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0])
p_out_limit_updated8 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total-out_VtF55_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0]-VtF55.shape[0])
p_out_limit_updated9 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total-out_VtF55_total-out_JunF_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0]-VtF55.shape[0]-JunF.shape[0])
p_out_limit_updated10 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total-out_VtF55_total-out_JunF_total-out_PromF_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0]-VtF55.shape[0]-JunF.shape[0]-PromF.shape[0])
p_out_limit_updated11 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total-out_VtF55_total-out_JunF_total-out_PromF_total-out_JunM_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0]-VtF55.shape[0]-JunF.shape[0]-PromF.shape[0]-JunM.shape[0])
p_out_limit_updated12 = (n_out_total-out_SenF_total-out_VtF35_total-out_SenM_total-out_VtM45_total-out_VtM35_total-out_VtF45_total-out_VtM55_total-out_VtF55_total-out_JunF_total-out_PromF_total-out_JunM_total-out_PromM_total)/(data.shape[0]-SenF.shape[0]-VtF35.shape[0]-SenM.shape[0]-VtM45.shape[0]-VtM35.shape[0]-VtF45.shape[0]-VtM55.shape[0]-VtF55.shape[0]-JunF.shape[0]-PromF.shape[0]-JunM.shape[0]-PromM.shape[0])

plt.figure()
plt.scatter(np.arange(13), [p_out_limit,p_out_limit_updated,p_out_limit_updated2,p_out_limit_updated3,
            p_out_limit_updated4,p_out_limit_updated5,p_out_limit_updated6,p_out_limit_updated7,
            p_out_limit_updated8,p_out_limit_updated9,p_out_limit_updated10,p_out_limit_updated11,
            p_out_limit_updated12], c=np.arange(13))
plt.title('Number of run out of %i seconds' % limit)
plt.xticks(range(13), ['-', 'SenF', 'VtF35', 'SenM', 'VtM45', 'VtM35', 'VtF45', 'VtM55', 'VtF55', 'JunF', 'PromF', 'JunM', 'PromM'], color='black', rotation=30)
for spine in plt.gca().spines.values():
    spine.set_visible(False)   
    
print('SenF, VtF35, SenM and VtM45 should be denied access to go down p(x>%i) from %.4f%% to %.4f%%' % (limit,100*p_out_limit,100*p_out_limit_updated4))


                