from turtle import color
import matplotlib.pyplot as plt
from matplotlib import lines
import math
import numpy as np
import community as co
import networkx as nx
import networkx.algorithms.community as nx_comm
from numpy import number
#from MBO_Network import  mbo_modularity_1_normalized_lf,mbo_modularity_2
#from MBO_Network import mbo_modularity_inner_step,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh
from graph_mbo.utils import get_fidelity_term,vector_to_labels,label_to_dict
import time
from MBO_signednetwork import mbo_modularity
from MBO_Network import mbo_modularity_1,mbo_modularity_1_normalized_lf,mbo_modularity_1_normalized_Qh,mbo_modularity_1_normalized_Lf_Qh,mbo_modularity_2,adj_to_laplacian_signless_laplacian


#accumulator = 0
#for _ in range(20):
#    reading = modularity_1_unnor_individual()
#    accumulator += reading
#average = accumulator / 20

#start_time = time.time()
#main()
#print("--- %s seconds ---" % (time.time() - start_time))

#noise_variable = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
#ARI_noise = [0.4767, 0.4760, 0.4742, 0.5592, 0.5324, 0.3894, 0.1725, 0.0075, 0.0024]
#plt.plot(noise_variable, ARI_noise)
#plt.title('N = 1500, K = 6, $\lambda$ = 0.10')
#plt.xlabel('$\eta$')
#plt.ylabel('ARI')
#plt.show()

G = nx.karate_club_graph()
adj_mat = nx.convert_matrix.to_numpy_matrix(G)

num_communities = [2, 3, 4, 5, 6, 7, 8]


#dt = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009]
#dt_list = []
#stride = 0.01
#number_quantity = 9
#for nummer in range(1, number_quantity+1):
#    dt_list.append(round(nummer*stride,2))
#print(dt_list)

#tol = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
#tol_list = []
#stride = 0.001
#number_quantity = 9
#for nummer in range(1, number_quantity+1):
#    tol_list.append(round(nummer*stride,2))
#print(dt_list)

modularity_1_unnor_list = []
modularity_1_unnor_list_06 = []
modularity_1_unnor_list_05 = []
modularity_1_nor_list = []
modularity_1_nor_list_06 = []
modularity_1_nor_list_05 = []
modularity_1_nor_Qh_list_1 = []
modularity_1_nor_Qh_list_06 = []
modularity_1_nor_Qh_list_05 = []
modularity_1_nor_Lf_Qh_list_1 = []
modularity_1_nor_Lf_Qh_list_06 = []
modularity_1_nor_Lf_Qh_list_05 = []
modularity_2_list = []
modularity_ori_list = []
modularity_1_inner_list_1 = []

average_1_unnor_list = []
average_1_unnor_05_list =[]
average_1_unnor_06_list =[]
average_2_nor_1_list =[]
average_1_nor_1_list =[]
average_1_nor_06_list =[]
average_1_nor_05_list =[]
average_1_nor_Qh_individual_1_list =[]
average_1_nor_Qh_individual_06_list =[]
average_1_nor_Qh_individual_05_list =[]
average_1_nor_Lf_Qh_individual_1_list =[]
average_1_nor_Lf_Qh_individual_06_list =[]
average_1_nor_Lf_Qh_individual_05_list =[]

# Parameter setting
dt_inner = 0.1
tol = 0
eta_1 = 1
eta_06 = 0.6
eta_05 = 0.5
eta_03 = 1.3
inner_step_count =3


for i in num_communities:
#    num_communities = 5
    m = 1 * i
    accumulator_1_unnor_individual = 0
    accumulator_1_unnor_individual_05 =0
    accumulator_1_unnor_individual_06 =0
    accumulator_2_nor_individual_1 =0
    accumulator_1_nor_individual_1 =0
    accumulator_1_nor_individual_06 =0
    accumulator_1_nor_individual_05 =0
    accumulator_1_nor_Qh_individual_1 =0
    accumulator_1_nor_Qh_individual_05 =0
    accumulator_1_nor_Qh_individual_06 =0
    accumulator_1_nor_Lf_Qh_individual_1 =0
    accumulator_1_nor_Lf_Qh_individual_06 =0
    accumulator_1_nor_Lf_Qh_individual_05 =0

    num_nodes_1,m_1, degree_1, target_size_1,null_model_eta_1,graph_laplacian_1, nor_graph_laplacian_1, signless_laplacian_null_model_1, nor_signless_laplacian_1 = adj_to_laplacian_signless_laplacian(adj_mat,i,m,eta_1,target_size=None)
    num_nodes_06,m_06, degree_06, target_size_06,null_model_eta_06,graph_laplacian_06, nor_graph_laplacian_06, signless_laplacian_null_model_06, nor_signless_laplacian_06 = adj_to_laplacian_signless_laplacian(adj_mat,i,m,eta_06,target_size=None)
    num_nodes_05,m_05, degree_05, target_size_05,null_model_eta_05,graph_laplacian_05, nor_graph_laplacian_05, signless_laplacian_null_model_05, nor_signless_laplacian_05 = adj_to_laplacian_signless_laplacian(adj_mat,i,m,eta_05,target_size=None)
    
    for _ in range(20):
    # mmbo 1 with unnormalized L_F and gamma=1
        u_1_unnor_individual,num_repeat_1_unnor = mbo_modularity_1(num_nodes_1,i, m_1,degree_1, graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)   
        u_1_unnor_individual_label = vector_to_labels(u_1_unnor_individual)
        u_1_unnor_individual_label_dict = label_to_dict(u_1_unnor_individual_label)
    #    #print('u_1_unnormalized label: ', u_1_unnor_individual_label)
        
        modularity_1_unnor_individual = co.modularity(u_1_unnor_individual_label_dict,G)
        modularity_1_unnor_list.append(modularity_1_unnor_individual)
        
        accumulator_1_unnor_individual += modularity_1_unnor_individual
    average_1_unnor = accumulator_1_unnor_individual / 20
    average_1_unnor_list.append(average_1_unnor)
    #print(average_list)

    #for _ in range(20):
    # mmbo 1 with unnormalized L_F and gamma = 0.5
    #    u_1_unnor_individual_05,num_repeat_1_unnor_05 = mbo_modularity_1(num_nodes_05,i, m_05,degree_05, graph_laplacian_05,signless_laplacian_null_model_05, 
    #                                                    tol, target_size_05,eta_05, eps=1)     
    #    u_1_unnor_individual_label_05 = vector_to_labels(u_1_unnor_individual_05)
    #    u_1_unnor_individual_label_dict_05 = label_to_dict(u_1_unnor_individual_label_05)
        
    #    modularity_1_unnor_individual_05 = co.modularity(u_1_unnor_individual_label_dict_05,G)
    #    modularity_1_unnor_list_05.append(modularity_1_unnor_individual_05)

    #    accumulator_1_unnor_individual_05 += modularity_1_unnor_individual_05
    #average_1_unnor_05 = accumulator_1_unnor_individual_05 / 20
    #average_1_unnor_05_list.append(average_1_unnor_05)

    #for _ in range(20):
    # mmbo 1 with unnormalized L_F and gamma = 0.6
    #    u_1_unnor_individual_06,num_repeat_1_unnor_06 = mbo_modularity_1(num_nodes_06,i, m_06,degree_06, graph_laplacian_06,signless_laplacian_null_model_06, 
    #                                                    tol, target_size_06,eta_06, eps=1)     
    #    u_1_unnor_individual_label_06 = vector_to_labels(u_1_unnor_individual_06)
    #    u_1_unnor_individual_label_dict_06 = label_to_dict(u_1_unnor_individual_label_06)
        
    #    modularity_1_unnor_individual_06 = co.modularity(u_1_unnor_individual_label_dict_06,G)
    #    modularity_1_unnor_list_06.append(modularity_1_unnor_individual_06)

    #    accumulator_1_unnor_individual_06 += modularity_1_unnor_individual_06
    #average_1_unnor_06 = accumulator_1_unnor_individual_06 / 20
    #average_1_unnor_06_list.append(average_1_unnor_06)


    for _ in range(20):
    # mmbo 2 with normalized & gamma = 1
        u_2_individual, num_repeat_2 = mbo_modularity_2(i, m, adj_mat, tol,eta_1,eps=1) 
        u_2_individual_label = vector_to_labels(u_2_individual)
        u_2_individual_label_dict = label_to_dict(u_2_individual_label)
        
        modularity_2_individual = co.modularity(u_2_individual_label_dict,G)
        modularity_2_list.append(modularity_2_individual)

        accumulator_2_nor_individual_1 += modularity_2_individual
    average_2_nor_1 = accumulator_2_nor_individual_1 / 20
    average_2_nor_1_list.append(average_2_nor_1)
    
    for _ in range(20):
    ## MMBO1 with normalized L_F and gamma=1
        u_1_nor_individual,num_repeat_1_nor = mbo_modularity_1_normalized_lf(num_nodes_1,i, m_1,degree_1, nor_graph_laplacian_1,signless_laplacian_null_model_1, 
                                                        tol, target_size_1,eta_1, eps=1)       
        u_1_nor_individual_label = vector_to_labels(u_1_nor_individual)
        u_1_nor_individual_label_dict = label_to_dict(u_1_nor_individual_label)
       
        modularity_1_nor_individual = co.modularity(u_1_nor_individual_label_dict,G)
        modularity_1_nor_list.append(modularity_1_nor_individual)

        accumulator_1_nor_individual_1 += modularity_1_nor_individual
    average_1_nor_1 = accumulator_1_nor_individual_1 / 20
    average_1_nor_1_list.append(average_1_nor_1)
    

    #for _ in range(20):
    ## MMBO1 with normalized L_F and gamma=0.5
    #    u_1_nor_individual_05,num_repeat_1_nor_05 = mbo_modularity_1_normalized_lf(num_nodes_05,i, m_05,degree_05, nor_graph_laplacian_05,signless_laplacian_null_model_05, 
    #                                                    tol, target_size_05,eta_05, eps=1)       
    #    u_1_nor_individual_label_05 = vector_to_labels(u_1_nor_individual_05)
    #    u_1_nor_individual_label_dict_05 = label_to_dict(u_1_nor_individual_label_05)
        
    #    modularity_1_nor_individual_05 = co.modularity(u_1_nor_individual_label_dict_05,G)
    #    modularity_1_nor_list_05.append(modularity_1_nor_individual_05)
        
    #    accumulator_1_nor_individual_05 += modularity_1_nor_individual_05
    #average_1_nor_05 = accumulator_1_nor_individual_05 / 20
    #average_1_nor_05_list.append(average_1_nor_05)
    

    #for _ in range(20):
    ## MMBO1 with normalized L_F and gamma=0.6
    #    u_1_nor_individual_06,num_repeat_1_nor_06 = mbo_modularity_1_normalized_lf(num_nodes_06,i, m_06,degree_06, nor_graph_laplacian_06,signless_laplacian_null_model_06, 
    #                                                    tol, target_size_06,eta_06, eps=1)     
    #    u_1_nor_individual_label_06 = vector_to_labels(u_1_nor_individual_06)
    #    u_1_nor_individual_label_dict_06 = label_to_dict(u_1_nor_individual_label_06)
        
    #    modularity_1_nor_individual_06 = co.modularity(u_1_nor_individual_label_dict_06,G)
    #    modularity_1_nor_list_06.append(modularity_1_nor_individual_06)
        
    #    accumulator_1_nor_individual_06 += modularity_1_nor_individual_06
    #average_1_nor_06 = accumulator_1_nor_individual_06 / 20
    #average_1_nor_06_list.append(average_1_nor_06)
    
    #for _ in range(20):
    ## MMBO1 with inner step & unnormalized L_F and gamma=1
        #u_inner_individual_1,num_repeat_inner = mbo_modularity_inner_step(i,m,adj_mat, dt_inner, tol,inner_step_count, gamma_1)
        #u_inner_individual_label_1 = vector_to_labels(u_inner_individual_1)
        #u_inner_individual_label_dict_1 = label_to_dict(u_inner_individual_label_1)
        #print(num_repeat_inner)

        #modularity_1_inner_individual_1 = co.modularity(u_inner_individual_label_dict_1,G)
        #modularity_1_inner_list_1.append(modularity_1_inner_individual_1)
        
        #accumulator_1_nor_individual_1 += modularity_1_inner_individual_1
    #average_1_nor_1 = accumulator_1_nor_individual_1 / 20
    #average_1_nor_1_list.append(average_1_nor_1)
    

    for _ in range(20):
        ## MMBO1 with normalized Q_H and gamma=1
        u_1_nor_Qh_individual_1,num_repeat_1_nor_Qh_1 = mbo_modularity_1_normalized_Qh(num_nodes_1,i, m_1,degree_1, graph_laplacian_1,nor_signless_laplacian_1, 
                                                        tol, target_size_1,eta_1, eps=1)     
        u_1_nor_Qh_individual_label_1 = vector_to_labels(u_1_nor_Qh_individual_1)
        u_1_nor_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Qh_individual_label_1)
        
        modularity_1_nor_Qh_individual_1 = co.modularity(u_1_nor_Qh_individual_label_dict_1,G)
        modularity_1_nor_Qh_list_1.append(modularity_1_nor_Qh_individual_1)
        
        accumulator_1_nor_Qh_individual_1 += modularity_1_nor_Qh_individual_1
    average_1_Qh_individual_1 = accumulator_1_nor_Qh_individual_1 / 20
    average_1_nor_Qh_individual_1_list.append(average_1_Qh_individual_1)
    

    #for _ in range(20):
    #    ## MMBO1 with normalized Q_H and gamma=0.6
    #    u_1_nor_Qh_individual_06,num_repeat_1_nor_Qh_06 = mbo_modularity_1_normalized_Qh(num_nodes_06,i, m_06,degree_06, graph_laplacian_06,nor_signless_laplacian_06, 
    #                                                    tol, target_size_06,eta_06, eps=1)       
    #    u_1_nor_Qh_individual_label_06 = vector_to_labels(u_1_nor_Qh_individual_06)
    #    u_1_nor_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Qh_individual_label_06)
        
    #    modularity_1_nor_Qh_individual_06 = co.modularity(u_1_nor_Qh_individual_label_dict_06,G)
    #    modularity_1_nor_Qh_list_06.append(modularity_1_nor_Qh_individual_06)

    #    accumulator_1_nor_Qh_individual_06 += modularity_1_nor_Qh_individual_06
    #average_1_Qh_individual_06 = accumulator_1_nor_Qh_individual_06 / 20
    #average_1_nor_Qh_individual_06_list.append(average_1_Qh_individual_06)

    #for _ in range(20):
    ## MMBO1 with normalized Q_H and gamma=0.5
    #    u_1_nor_Qh_individual_05,num_repeat_1_nor_Qh_05 = mbo_modularity_1_normalized_Qh(num_nodes_05,i, m_05,degree_05, graph_laplacian_05,nor_signless_laplacian_05, 
    #                                                    tol, target_size_05,eta_05, eps=1)   
    #    u_1_nor_Qh_individual_label_05 = vector_to_labels(u_1_nor_Qh_individual_05)
    #    u_1_nor_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Qh_individual_label_05)
        
    #    modularity_1_nor_Qh_individual_05 = co.modularity(u_1_nor_Qh_individual_label_dict_05,G)
    #    modularity_1_nor_Qh_list_05.append(modularity_1_nor_Qh_individual_05)

    #    accumulator_1_nor_Qh_individual_05 += modularity_1_nor_Qh_individual_05
    #average_1_Qh_individual_05 = accumulator_1_nor_Qh_individual_05 / 20
    #average_1_nor_Qh_individual_05_list.append(average_1_Qh_individual_05)

    for _ in range(20):
    ## MMBO1 with normalized L_F & Q_H and gamma=1
        u_1_nor_Lf_Qh_individual_1,num_repeat_1_nor_Lf_Qh_1 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_1,i, m_1,degree_1, nor_graph_laplacian_1,nor_signless_laplacian_1, 
                                                        tol, target_size_1,eta_1, eps=1)     
        u_1_nor_Lf_Qh_individual_label_1 = vector_to_labels(u_1_nor_Lf_Qh_individual_1)
        u_1_nor_Lf_Qh_individual_label_dict_1 = label_to_dict(u_1_nor_Lf_Qh_individual_label_1)
        
        modularity_1_nor_Lf_Qh_individual_1 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_1,G)
        modularity_1_nor_Lf_Qh_list_1.append(modularity_1_nor_Lf_Qh_individual_1)

        accumulator_1_nor_Lf_Qh_individual_1 += modularity_1_nor_Lf_Qh_individual_1
    average_1_nor_Lf_Qh_individual_1 = accumulator_1_nor_Lf_Qh_individual_1 / 20
    average_1_nor_Lf_Qh_individual_1_list.append(average_1_nor_Lf_Qh_individual_1)

    #for _ in range(20):
    ## MMBO1 with normalized L_F & Q_H and gamma=0.6
    #    u_1_nor_Lf_Qh_individual_06,num_repeat_1_nor_Lf_Qh_06 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_06,i, m_06,degree_06, nor_graph_laplacian_06,nor_signless_laplacian_06, 
    #                                                    tol, target_size_06,eta_06, eps=1)       
    #    u_1_nor_Lf_Qh_individual_label_06 = vector_to_labels(u_1_nor_Lf_Qh_individual_06)
    #    u_1_nor_Lf_Qh_individual_label_dict_06 = label_to_dict(u_1_nor_Lf_Qh_individual_label_06)
        
    #    modularity_1_nor_Lf_Qh_individual_06 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_06,G)
    #    modularity_1_nor_Lf_Qh_list_06.append(modularity_1_nor_Lf_Qh_individual_06)

    #    accumulator_1_nor_Lf_Qh_individual_06 += modularity_1_nor_Lf_Qh_individual_06
    #average_1_nor_Lf_Qh_individual_06 = accumulator_1_nor_Lf_Qh_individual_06 / 20
    #average_1_nor_Lf_Qh_individual_06_list.append(average_1_nor_Lf_Qh_individual_06)

    #for _ in range(20):
        ## MMBO1 with normalized L_F & Q_H and gamma=0.5
    #    u_1_nor_Lf_Qh_individual_05,num_repeat_1_nor_Lf_Qh_05 = mbo_modularity_1_normalized_Lf_Qh(num_nodes_05,i, m_05,degree_05, nor_graph_laplacian_05,nor_signless_laplacian_05, 
    #                                                    tol, target_size_05,eta_05, eps=1)        
    #    u_1_nor_Lf_Qh_individual_label_05 = vector_to_labels(u_1_nor_Lf_Qh_individual_05)
    #    u_1_nor_Lf_Qh_individual_label_dict_05 = label_to_dict(u_1_nor_Lf_Qh_individual_label_05)
        
    #    modularity_1_nor_Lf_Qh_individual_05 = co.modularity(u_1_nor_Lf_Qh_individual_label_dict_05,G)
    #    modularity_1_nor_Lf_Qh_list_05.append(modularity_1_nor_Lf_Qh_individual_05)

    #    accumulator_1_nor_Lf_Qh_individual_05 += modularity_1_nor_Lf_Qh_individual_05
    #average_1_nor_Lf_Qh_individual_05 = accumulator_1_nor_Lf_Qh_individual_05 / 20
    #average_1_nor_Lf_Qh_individual_05_list.append(average_1_nor_Lf_Qh_individual_05)


#plt.figure(figsize=(8,8))
plt.plot(num_communities, average_1_unnor_list,'-',color='C0', label = "MMBO1 with $\eta=1$ and unnormalized $L_F$")
#plt.plot(num_communities, average_1_unnor_06_list, '-',color='C1', label = "MMBO1 with $\eta=0.6$ and unnormalized $L_F$")
#plt.plot(num_communities, average_1_unnor_05_list,'-',color='C2', label = "MMBO1 with $\eta=0.5$ and unnormalized $L_F$")
plt.plot(num_communities, average_1_nor_1_list,'--',color='C3', label = "MMBO1 with $\eta=1$ and $L_{F_{sym}}$")
#plt.plot(num_communities, average_1_nor_06_list, '--',color='C4', label = "MMBO1 with $\eta=0.6$ and $L_{F_{sym}}$")
#plt.plot(num_communities, average_1_nor_05_list, '--',color='C5', label = "MMBO1 with $\eta=0.5$ and $L_{F_{sym}}$")
plt.plot(num_communities, average_1_nor_Qh_individual_1_list, '-.',color='C6', label = "MMBO1 with $\eta=1$ and $Q_{H_{sym}}$")
#plt.plot(num_communities, average_1_nor_Qh_individual_06_list, '-.',color='C7', label = "MMBO1 with $\eta=0.6$ and $Q_{H_{sym}}$")
#plt.plot(num_communities, average_1_nor_Qh_individual_05_list, '-.',color='C8', label = "MMBO1 with $\eta=0.5$ and $Q_{H_{sym}}$")
plt.plot(num_communities, average_1_nor_Lf_Qh_individual_1_list, ':',color='C9', label = "MMBO1 with $\eta=1$ and $L_{F_{sym}}$, $Q_{H_{sym}}$")
#plt.plot(num_communities, average_1_nor_Lf_Qh_individual_06_list,':',color='C10', label = "MMBO1 with $\eta=0.6$ and $L_{F_{sym}}$, $Q_{H_{sym}}$")
#plt.plot(num_communities, average_1_nor_Lf_Qh_individual_05_list, ':',color='C11',label = "MMBO1 with $\eta=0.5$ and $L_{F_{sym}}$, $Q_{H_{sym}}$")
plt.plot(num_communities, average_2_nor_1_list,':',color='C12', label = "MMBO2 with $\eta=1$ and $L_{{B^+}_{sym}}$, $Q_{{B^-}_{sym}}$")
#plt.plot(num_communities, modularity_1_unnor_list, label = "MMBO 1 with innerstep & normalized #L_F")
plt.title('$m=K, \epsilon=0, \gamma=1$')
plt.xlabel('Number of clusters')
plt.ylabel('Modularity scores')
x_axis = num_communities
xint = range(min(x_axis ), math.ceil(max(x_axis ))+1)
plt.xticks(xint)
plt.legend()
plt.show()

#print("--- %.2f seconds ---" % (time.time() - start_time))