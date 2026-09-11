# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:09:23 2019

@author: yaginuma
"""
import GaussianProcessMultiDim
import numpy as np
import matplotlib.pyplot as plt
import utils

class Trajectory_Generator():
    def __init__(self,dim,numclass,dirname):
        self.dim = dim
        self.numclass = numclass
        self.dir = dirname
        self.gps = [GaussianProcessMultiDim.GPMD(self.dim) for i in range(self.numclass)]
        self.segm_in_class = [[] for i in range(self.numclass)]
      
    def load_data(self):
        for c in range(self.numclass):
            filename = self.dir + "class%03d.npy" % c
            self.segm_in_class[c] = np.load(filename, allow_pickle=True)
            
            if self.dim == 2:
                pass
            if self.dim == 3:
                for i in range(len(self.segm_in_class[c])):
                    zeros = np.zeros((1,len(self.segm_in_class[c][i])))
                    self.segm_in_class[c][i] = np.hstack((self.segm_in_class[c][i], zeros.T))
    
    #エキスパートポリシーの軌道を生成
    def generate_expert_trajectory(self):
        self.load_data()
        params = self.get_mean_trajectory()
        up_data = self.upsampling(params)
        d_data = self.decoder(up_data)
        return d_data
    
    #データから軌道の平均と分散を取り出す
    def get_mean_trajectory(self):
        #params[c] : [[[mu_x1,sigma_x1]...[mu_xn,sigma_xn]],[[mu_y1,sigma_y1]...[mu_yn,sigma_yn]]]
        params = [[] for i in range(self.numclass)]
        for c in range(self.numclass):
            datay = []
            datax = []
            for s in self.segm_in_class[c]:
                datay += [ y for y in s ]
                datax += range(len(s))
            self.gps[c].learn( datax, datay )
            #mu,sigma = self.gps[c].plot([ max(datax)*i/100.0 for i in range(101) ],returning=True)
            mu,sigma = self.gps[c].plot([ i for i in np.linspace(0,max(datax),1000) ],returning=True)
            for i in range(self.dim):
                params[c].append(np.vstack((mu[i],sigma[i])).T)
        plt.clf()
        return params
    
    #2つのガウス過程を重ね合わせて新しい軌道を生成
    def merge_trajectory(self, params1, params2):
        new_means = []
        new_sigmas = []
        
        for p1, p2 in zip(params1, params2):
            m1 = p1[0]
            s1 = p1[1]
            m2 = p2[0]
            s2 = p2[1]
        
            mu = (s2*m1 + s1*m2) / ( s1 + s2 )
            s = s1 * s2 /(s1+s2)
        
            new_means.append(mu)
            new_sigmas.append(s)
            
        return new_means, new_sigmas
    
    #シミュレータ上で使えるようにアップサンプリング(データ点を増やすだけ)
    def upsampling(self, data, multiple=100):
        up_data = []
        for c in range(len(data)):
            up_data_segm = []
            for i in range(len(data[c])):
                sampled_segm = []
                temp = data[c][i]
                for j in range(len(data[c][i][0])):
                    x = temp[:,j]
                    up_data_segm_x = []
                    for k in range(len(x)-1):
                        up_x = np.linspace(x[k],x[k+1],multiple,endpoint=False)
                        up_data_segm_x.extend(up_x)
                    sampled_segm.append(up_data_segm_x)
                up_data_segm.append(np.array(sampled_segm).T)
            up_data.append(up_data_segm)
            
        return up_data

    #物体の初期位置に応じた軌道を生成
    def dummy_trajectory(self, pos, segm_len, sigma_scale=1000.0):
        #注：params2 = [[[mu_x1 sigma_x1]...],[[mu_z1 simga_z1]...],[[mu_pitch1 sigma_pitch1]...]]
        params2 = []
        rng = [ segm_len * i / ((segm_len-1)*100) for i in range(((segm_len-1)*100))]
        for data in range(len(pos)):
            tmp = []
            for r in rng:
                tmp.append([pos[data], r/sigma_scale])
            params2.append(tmp)
        params2 = np.array(params2)
        
        return rng, params2
    
    #物体の補正位置に応じた軌道を生成
    def end_dummy_trajectory(self, pos, segm_len, sigma_scale=2000.0):
        #注：params2 = [[[mu_x1 sigma_x1]...],[[mu_z1 simga_z1]...],[[mu_pitch1 sigma_pitch1]...]]
        params2 = []
        rng = [ segm_len * i / ((segm_len-1)*100) for i in range(((segm_len-1)*100))]
        for data in range(len(pos)):
            tmp = []
            for r in rng:
                tmp.append([pos[data], r/(sigma_scale)])
            tmp = tmp[::-1]
            params2.append(tmp)
        params2 = np.array(params2)
        
        return rng, params2
    
    #分節化したデータを元のスケールに戻す
    def decoder(self, data):
        num_decodes = []
        mu_max = [0.35582185, 0.53946611, 1.9678888]
        mu_min = [-0.14860694, -0.11744604, -0.49280819]
        
        for c in range(len(data)):
            xs = []
            for i in range(len(data[c])):
                xxs = []
                for j in range(len(data[c][i])):
                    x = (data[c][i][j][0]+1)*(mu_max[i]-mu_min[i])/2+mu_min[i]
                    xxs.append([x,data[c][i][j][1]])
                xs.append(xxs)
            num_decodes.append(xs)
        return num_decodes
        
    

if __name__ == "__main__":
    dim = 2
    numclass = 12
    dirname = "C:/Users/k_yaginuma/Documents/git-repo/yaginuma-tracker/mm_2dim_max15_min5_ave15_segm12_data30_c/"
    
    tg = Trajectory_Generator(dim,numclass)
    tg.load_data(dirname)
    params = tg.get_mean_orbit()
    up_data = tg.upsampling(params)
    d_data = tg.decoder(up_data)
    print(np.array(d_data).shape)
    print(d_data[0][0])
#    for c in range(numclass):
#        np.savetxt("C:/Users/yaginuma/Documents/git_repo/yaginuma-tracker/test/class%03d.txt" % c, np.array([up_data[c][0][:,0],up_data[c][1][:,0]]).T)
#    for c in range(numclass):   
#        np.save("C:/Users/yaginuma/Documents/git_repo/yaginuma-tracker/test/class%03d.npy" % c,up_data[c])
#    
    rng, params2 = tg.dummy_trajectory(pos=[0,0,0], segm_len=len(params[1][0]), sigma_scale=1000.0)
    utils.plot(rng, up_data[1][0][:,0], up_data[1][0][:,1])
    plt.show()
    utils.plot(rng, params2[0][:,0], params2[0][:,1])
    plt.show()
    mu1, sigma1 = tg.merge_trajectory(up_data[1][0],params2[0])
    mu2, sigma2 = tg.merge_trajectory(up_data[1][1],params2[1])
    old_orbit1 = np.array([mu1,sigma1]).T
    old_orbit2 = np.array([mu2,sigma2]).T
    print(old_orbit1)
    rng, params3 = tg.end_dummy_trajectory(pos=[0,0,0], segm_len=len(params[1][0]), sigma_scale=3000.0)
    utils.plot(rng, params3[0][:,0], params3[0][:,1])
    plt.show()
    n_mu1, n_sigma1 = tg.merge_trajectory(old_orbit1, params3[0])
    n_mu2, n_sigma2 = tg.merge_trajectory(old_orbit2, params3[1])
    plt.plot(mu1,mu2)
    plt.show()
    utils.plot(rng, mu1, sigma1 )
    plt.show()
    utils.plot(rng, mu2, sigma2 )
    plt.show()
    utils.plot(rng, n_mu1, n_sigma1 )
    plt.show()
    utils.plot(rng, n_mu2, n_sigma2 )
    plt.show()
    
    for c in range(numclass):
        plt.plot(d_data[c][1][:,0],d_data[c][0][:,0], label="class {}".format(c))
    plt.legend()
    plt.xlim([0.6,-0.2])
    plt.ylim([0.4,-0.2])
    plt.savefig("mean_trajectory.svg", format="svg",dpi=1200)
    #np.save("C:/Users/yaginuma/Desktop/GP-HSMM/mm_decoder/class%03d.npy" % c, num_decode)
