#description of the programm
import matplotlib.pyplot as plt
import numpy as np
from ap_utils5Copy1 import (
    APSignal,  entrenar_modelo, evaluar_modelo, HP_model, compare_hp, compare_hp_old, evaluate_single_model, plot_pred, pred_lstm_gru, pred_reg, overall_loss, apds_pred, to_log, evaluate_single_model_sequential)

###############################################################
minute = 0
t_tol = 20   # dt = 0.05 --> 1 de cada 20 muestras --> escala en msec 
v_vol = -1
fpath = "data/min0_all.npy"  #path in olivia?

#fresults = "alpha_results.txt"
fresults = "stepwise_results.txt"
#fresults = "olivia_results.txt"
fout = open(fresults,"a")
to_log('------------------------------------------', fout)

thresholds ={ 0: -84, 3: -74, 5: -67, 9:-65, 13:-66}    

#ap = APSignal(fpath, t_tol, v_vol, thresholds[minute],  show=False, mode=1, normalize=False)

#model = HP_model(ap, mode='gru', lback = 64, hp=10, nclusters=10, alpha=0.5) 

#X_train, Y_train, X_test, Y_test = model.build_data_set()
#print(X_train.shape)
#print(Y_train.shape)
#print(X_test.shape)
#print(Y_test.shape)
#print(ap.size)

#ap.plot_apds() #plot the apds

#compare_hp_old(ap, fout, epochs=2, lback=128, hp=10, batch_size=64, nclusters=10)
#compare_hp(ap, fout, epochs=20, lback=64, hp=8, batch_size=32, plot_pulses=False, plot_accuracy=False, plot_loss=False, nclusters=10)

ap = APSignal(fpath, t_tol, v_vol, thresholds[minute],  show=False, mode=0, normalize=False)
#results = evaluate_single_model(ap, fout, mode='lstm', lback=64, hp=8, plot_loss=False, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=True, nclusters=10)
results2 = evaluate_single_model_sequential(ap, fout, mode='reg', lback=64, hp=16, plot_loss=False, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=True, nclusters=10, steps=1) #steps is the stepsize

#olivia
# nclusters = [10, 20]
# hps = [8, 16, 32]
# modes = [0, 1]
# for nc in nclusters:

#     for mod in modes:
#         ap = APSignal(fpath, t_tol, v_vol, thresholds[minute],  show=False, mode=mod, normalize=False)
#         for hp in hps:
#             #print(nc, hp, mod)
#             results = evaluate_single_model(ap, fout, mode='reg', lback=64, hp=hp, plot_loss=False, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=True, nclusters=nc)

# lbs = [32, 64, 128]
# for lb in lbs:
#     for nc in nclusters:
#         for mod in modes:
#             #print(lb, nc, mod)
#             ap = APSignal(fpath, t_tol, v_vol, thresholds[minute],  show=False, mode=mod, normalize=False)
#             results = evaluate_single_model(ap, fout, mode='reg', lback=lb, hp=8, plot_loss=False, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=True, nclusters=nc)
###            


#stepwise prediction loop 
#immer mode=0, der rest macht wenig Sinn
#teilweise sind verschiedene Architecturen nötig, deswegen nur teilweise automatisiert
# modes = ['lstm', 'gru'] #gru und lstm haben sowieso die gleiche Architecture
# steps = [1, 2, 4]
# for mode in modes:
#     for step in steps:
#         results2 = evaluate_single_model_sequential(ap, fout, mode=mode, lback=64, hp=8, plot_loss=False, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=False, nclusters=10, steps=step) #steps is the stepsize


#results2 = evaluate_single_model_sequential(ap, fout, mode='lstm', lback=64, hp=32, plot_loss=True, epochs=1000, batch_size=32, plot_accuracy=False, plot_pulses=False, nclusters=10, steps=4) #steps is the stepsize


#print("Model Type:", results['model_type'])
#print("Prediction Error:", results['error'])
#print("Overall Error:", results['overall_error'])


