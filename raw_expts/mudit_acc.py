
import numpy as np
from numpy import  inf
import os
import sys
import scipy
from scipy import signal

main_folder = sys.argv[1] #'eusip_40ms_time'
main_model = sys.argv[2]#'awgn_40ms_time_multiscale_cnn_lstm_model_context_1_ptdb'

threshold = float(sys.argv[3]) #0.5
db_list = ['clean','20','10','5','0']

f_ref = 8.1

print('\n')
print('Main model : ',main_folder+'/'+main_model)
print('\n')
for db in db_list:

  folder = '../'+main_folder+'/training_scripts/models/'+main_model+'/results/'
  path = folder+'predict_PTDB_TUG_'+db+'_st.txt'

  truth = np.loadtxt('PTDB_TUG_ground_truth_of_eusip_exps_40ms.txt',delimiter=',')
  predict = np.loadtxt(path,delimiter=',')

  #predict = scipy.signal.medfilt(predict,5)# adding median filter
  truth = 12*(np.log2(truth/f_ref))#converting hz to semitones

  truth[truth == -inf] = 0
  truth[truth<=30] = 0

  predict[predict<30] = 0# already in semitones

  #----------------------------------------------------------
  u2u_frames =0;
  total = len(truth)

  truth_binary = np.sign(truth)
  pred_binary = np.sign(predict)

  voiced_frame_predict = np.sum(np.logical_and(truth_binary,pred_binary))
  voiced_frame_original = np.sum(truth_binary)
  VRR = (voiced_frame_predict/voiced_frame_original)*100

  abs_error = []
  for j in range(total):
    if (truth_binary[j]==1):
      if(pred_binary[j]==1):
        abs_error = np.append(abs_error,np.abs(truth[j]-predict[j]))


  abs_error1 = np.ones(len(abs_error))
  abs_error1[abs_error>threshold] = 0

  RPA = 100*(np.sum(abs_error1)/voiced_frame_original)

  VRR = str(round(VRR, 2))
  RPA = str(round(RPA, 2))


  if(db=='clean'):
      space = ''
  elif(int(db)>9):
      space = '   '
  else:
      space = '    '
  print(db,space,': VRR = ',VRR,' RPA = ',RPA)
