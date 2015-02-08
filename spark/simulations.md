
## Demonstration of debiasing


    # check Spark context
    sc




    <pyspark.context.SparkContext at 0x7ff50e2dd510>




    # run this first!
    f = open('/root/donuts/spark/sparkDonuts.py', 'r')
    exec(f.read())
    f.close()

# Table of Contents

 * [Introduction](#intro): An overview of the sparse fasicle model, and how we
use Earth Mover's Distance to easure error
 * [Part I: Example](#part1): A demonstration of how non-Gaussian noise biases
NNLS estimates, and how knowledge of the noise can be used to debias our
estimate
 * [Part II: Simulations](#part2) : Quantify the possible gain from debiasing
using simulations

<a id="intro"/>

## Introduction

###Consider a voxel with three fiber populations:


    pylab.rcParams['figure.figsize'] = (7.0, 7.0) # graphical parameters
    vox_dirs = normalize_rows(npr.randn(3, 3))
    print("True fiber configuration:")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for ii in range(3):
        ax.plot(np.array([-vox_dirs[ii, 0], vox_dirs[ii, 0]]), \
                np.array([-vox_dirs[ii, 1], vox_dirs[ii, 1]]), \
                np.array([-vox_dirs[ii, 2], vox_dirs[ii, 2]]))

    True fiber configuration:



![png](simulations_files/simulations_8_1.png)


###The noiseless diffusion signal for each fiber population looks like a "donut"


    pylab.rcParams['figure.figsize'] = (20.0, 6.0)
    vizgrid = georandsphere(4, 3) # grid for visualization
    fig = plt.figure()
    ax=[None]*3
    bvalue = 1.0 # scan parameters
    lambdas = [4.0, 0.0, 0.0] # tensor parameters
    mus = np.exp(-bvalue * lambdas[2]) * ste_tan_kappa(np.sqrt(bvalue * (lambdas[0]-lambdas[1])) * vox_dirs, vizgrid)
    ax[0] = fig.add_subplot(131, projection='3d')
    ax[0].set_title('Fiber pop 1')
    ax[1] = fig.add_subplot(132, projection='3d')
    ax[1].set_title('Fiber pop 2')
    ax[2] = fig.add_subplot(133, projection='3d')
    ax[2].set_title('Fiber pop 3')
    for ii in range(3):
        ax[ii].scatter(vizgrid[:, 0]*mus[:, ii], \
                       vizgrid[:, 1]*mus[:, ii], \
                       vizgrid[:, 2]*mus[:, ii], \
                       alpha = 0.5)
        ax[ii].set_xlim3d(-1, 1)
        ax[ii].set_ylim3d(-1, 1)
        ax[ii].set_zlim3d(-1, 1)


![png](simulations_files/simulations_10_0.png)


###The observed diffusion signal is a weighted combination of the individual
"donuts", plus noise


    pylab.rcParams['figure.figsize'] = (10.0, 10.0)
    w0 = np.ones(3)/3.0 # relative proporties of fiber populations
    mu0 = np.squeeze(np.dot(mus, w0)) # noiseless signal
    df = 64 # twice the number of coils
    sigma0 = 0.01/sqrt(df) # noise per coil
    y = rvs_ncx2(df, mu0, sigma=sigma0) # observed magnitude
    muhat = np.sqrt(y) # naive estimate of signal: square root of magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vizgrid[:, 0]*muhat, \
               vizgrid[:, 1]*muhat, \
               vizgrid[:, 2]*muhat, \
               alpha = 0.5)
    ax.set_xlim3d(-.7, .7)
    ax.set_ylim3d(-.7, .7)
    ax.set_zlim3d(-.7, .7)
    print("Noisy signal")

    Noisy signal



![png](simulations_files/simulations_12_1.png)


###Of course, in practice the SNR is worse and we have fewer measurement
directions


    pylab.rcParams['figure.figsize'] = (10.0, 10.0)
    bvecs = geosphere(3) # measurement directions
    print('Number of bvecs: ' + str(np.shape(bvecs)[0]))
    w0 = np.ones(3)/3.0 # relative proporties of fiber populations
    mu0 = np.exp(-bvalue * lambdas[2]) * \
        np.squeeze(np.dot(ste_tan_kappa(np.sqrt(bvalue * (lambdas[0]-lambdas[1])) * vox_dirs, bvecs), w0)) # noiseless signal
    df = 64 # twice the number of coils
    sigma0 = 0.1/sqrt(df) # noise per coil
    y = rvs_ncx2(df, mu0, sigma=sigma0) # observed magnitude
    muhat = np.sqrt(y) # naive estimate of signal: square root of magnitude
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(bvecs[:, 0]*muhat, \
               bvecs[:, 1]*muhat, \
               bvecs[:, 2]*muhat, \
               alpha = 0.5)
    plotscale = .7
    ax.set_xlim3d(-plotscale, plotscale)
    ax.set_ylim3d(-plotscale, plotscale)
    ax.set_zlim3d(-plotscale, plotscale)
    print("Realistic signal")

    Number of bvecs: 92
    Realistic signal



![png](simulations_files/simulations_14_1.png)


###We can estimate the signal using NNLS, a model which implicitly assumes
gaussian noise.


    sgrid = georandsphere(5,2)
    sgrid = sgrid[sgrid[:, 2] > 0, :]
    amat = np.exp(-bvalue * lambdas[2]) * \
        ste_tan_kappa(np.sqrt(bvalue * (lambdas[0]-lambdas[1])) * sgrid, bvecs) # regression matrix
    bt_nnls = spo.nnls(amat, muhat)[0]
    y0 = mu0**2 + df * sigma0**2
    muhat_nnls = np.squeeze(np.dot(amat, bt_nnls))
    sigmahat2_nnls = sum((muhat- muhat_nnls)**2)/len(muhat)
    yhat_nnls = muhat_nnls**2 + sigmahat2_nnls
    dirs_nnls = sgrid[bt_nnls > 0, :]
    wts_nnls = divsum(bt_nnls[bt_nnls > 0])
    mse_nnls 

###NNLS does well at predicting the expected signal


    print("MSE = ")
    plt.scatter(y0, yhat_nnls)




    <matplotlib.collections.PathCollection at 0x7ff4fa8fd050>




![png](simulations_files/simulations_18_1.png)



    

<a id="part1"/>

## Part I. Example


    


    

<a id="part2"/>

## Part II. Simulations


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    


    import subprocess


    subprocess.check_output('git -C /root/donuts add /root/donuts/spark/*.ipynb', shell=True)




    ''




    subprocess.check_output('git -C /root/donuts commit -a -m "commit inside python"', shell=True)




    '[master baea11a] commit inside python\n Committer: Charles EC2 <root@ip-172-31-20-178.us-west-2.compute.internal>\nYour name and email address were configured automatically based\non your username and hostname. Please check that they are accurate.\nYou can suppress this message by setting them explicitly:\n\n    git config --global user.name "Your Name"\n    git config --global user.email you@example.com\n\nAfter doing this, you may fix the identity used for this commit with:\n\n    git commit --amend --reset-author\n\n 1 file changed, 663 insertions(+)\n create mode 100644 spark/simulations.ipynb\n'




    subprocess.check_output('bash /root/gitcomp.sh', shell=True)




    'upload: ../.git/description to s3://braindatatest/donuts/.git/description\nCompleted 1 of 104 part(s) with 101 file(s) remaining\rupload: ../.git/ORIG_HEAD to s3://braindatatest/donuts/.git/ORIG_HEAD\nCompleted 2 of 104 part(s) with 100 file(s) remaining\rupload: ../.git/FETCH_HEAD to s3://braindatatest/donuts/.git/FETCH_HEAD\nCompleted 3 of 104 part(s) with 99 file(s) remaining\rupload: ../.git/COMMIT_EDITMSG to s3://braindatatest/donuts/.git/COMMIT_EDITMSG\nCompleted 4 of 104 part(s) with 98 file(s) remaining\rupload: ../.git/HEAD to s3://braindatatest/donuts/.git/HEAD\nCompleted 5 of 104 part(s) with 97 file(s) remaining\rupload: ../.git/hooks/post-update.sample to s3://braindatatest/donuts/.git/hooks/post-update.sample\nCompleted 6 of 104 part(s) with 96 file(s) remaining\rupload: ../.git/hooks/commit-msg.sample to s3://braindatatest/donuts/.git/hooks/commit-msg.sample\nCompleted 7 of 104 part(s) with 95 file(s) remaining\rupload: ../.git/hooks/applypatch-msg.sample to s3://braindatatest/donuts/.git/hooks/applypatch-msg.sample\nCompleted 8 of 104 part(s) with 94 file(s) remaining\rupload: ../.git/config to s3://braindatatest/donuts/.git/config\nCompleted 9 of 104 part(s) with 93 file(s) remaining\rupload: ../.git/hooks/pre-applypatch.sample to s3://braindatatest/donuts/.git/hooks/pre-applypatch.sample\nCompleted 10 of 104 part(s) with 92 file(s) remaining\rupload: ../.git/hooks/pre-commit.sample to s3://braindatatest/donuts/.git/hooks/pre-commit.sample\nCompleted 11 of 104 part(s) with 91 file(s) remaining\rupload: ../.git/hooks/pre-push.sample to s3://braindatatest/donuts/.git/hooks/pre-push.sample\nCompleted 12 of 104 part(s) with 90 file(s) remaining\rupload: ../.git/hooks/prepare-commit-msg.sample to s3://braindatatest/donuts/.git/hooks/prepare-commit-msg.sample\nCompleted 13 of 104 part(s) with 89 file(s) remaining\rupload: ../.git/hooks/pre-rebase.sample to s3://braindatatest/donuts/.git/hooks/pre-rebase.sample\nCompleted 14 of 104 part(s) with 88 file(s) remaining\rupload: ../.git/hooks/update.sample to s3://braindatatest/donuts/.git/hooks/update.sample\nCompleted 15 of 104 part(s) with 87 file(s) remaining\rupload: ../.git/index to s3://braindatatest/donuts/.git/index\nCompleted 16 of 104 part(s) with 86 file(s) remaining\rupload: ../.git/logs/refs/heads/master to s3://braindatatest/donuts/.git/logs/refs/heads/master\nCompleted 17 of 104 part(s) with 85 file(s) remaining\rupload: ../.git/info/exclude to s3://braindatatest/donuts/.git/info/exclude\nCompleted 18 of 104 part(s) with 84 file(s) remaining\rupload: ../.git/logs/refs/remotes/origin/HEAD to s3://braindatatest/donuts/.git/logs/refs/remotes/origin/HEAD\nCompleted 19 of 104 part(s) with 83 file(s) remaining\rupload: ../.git/logs/refs/remotes/origin/master to s3://braindatatest/donuts/.git/logs/refs/remotes/origin/master\nCompleted 20 of 104 part(s) with 82 file(s) remaining\rupload: ../.git/logs/HEAD to s3://braindatatest/donuts/.git/logs/HEAD\nCompleted 21 of 104 part(s) with 81 file(s) remaining\rupload: ../.git/objects/03/89c14854a12042fc7e71723bb50ed62e814b0e to s3://braindatatest/donuts/.git/objects/03/89c14854a12042fc7e71723bb50ed62e814b0e\nCompleted 22 of 104 part(s) with 80 file(s) remaining\rupload: ../.git/objects/13/0e9a7e8c7a4a83639137e6f8746d4851dd2260 to s3://braindatatest/donuts/.git/objects/13/0e9a7e8c7a4a83639137e6f8746d4851dd2260\nCompleted 23 of 104 part(s) with 79 file(s) remaining\rupload: ../.git/objects/33/fad5b5cf6c25f8cbd3e93f729b05d4454bbdb7 to s3://braindatatest/donuts/.git/objects/33/fad5b5cf6c25f8cbd3e93f729b05d4454bbdb7\nCompleted 24 of 104 part(s) with 78 file(s) remaining\rupload: ../.git/objects/53/a5d27efa8a7e23c29b6a6c02615f39bfab677d to s3://braindatatest/donuts/.git/objects/53/a5d27efa8a7e23c29b6a6c02615f39bfab677d\nCompleted 25 of 104 part(s) with 77 file(s) remaining\rupload: ../.git/objects/46/b0310e6be81afe13646de033a7504d83fa8d60 to s3://braindatatest/donuts/.git/objects/46/b0310e6be81afe13646de033a7504d83fa8d60\nCompleted 26 of 104 part(s) with 76 file(s) remaining\rupload: ../.git/objects/a0/504520dfb4515020b7c3c79f2d8c3a7b00a1d7 to s3://braindatatest/donuts/.git/objects/a0/504520dfb4515020b7c3c79f2d8c3a7b00a1d7\nCompleted 27 of 104 part(s) with 75 file(s) remaining\rupload: ../.git/objects/4a/81ad44058c47e28b405fd0e03c3687e49dea0c to s3://braindatatest/donuts/.git/objects/4a/81ad44058c47e28b405fd0e03c3687e49dea0c\nCompleted 28 of 104 part(s) with 74 file(s) remaining\rupload: ../.git/objects/a0/bfde40e6d8045923a072961e86eecfeb555254 to s3://braindatatest/donuts/.git/objects/a0/bfde40e6d8045923a072961e86eecfeb555254\nCompleted 29 of 104 part(s) with 73 file(s) remaining\rupload: ../.git/objects/a9/3586b6d66537c3c7df162e2f0ee94aaabc2937 to s3://braindatatest/donuts/.git/objects/a9/3586b6d66537c3c7df162e2f0ee94aaabc2937\nCompleted 30 of 104 part(s) with 72 file(s) remaining\rupload: ../.git/objects/85/1e5adedb6404f6120adbeac024a891fa287e9b to s3://braindatatest/donuts/.git/objects/85/1e5adedb6404f6120adbeac024a891fa287e9b\nCompleted 31 of 104 part(s) with 71 file(s) remaining\rupload: ../.git/objects/ba/ea11a857e5b6a36e9e7283f0c0b7d7f0eb3c87 to s3://braindatatest/donuts/.git/objects/ba/ea11a857e5b6a36e9e7283f0c0b7d7f0eb3c87\nCompleted 32 of 104 part(s) with 70 file(s) remaining\rupload: ../.git/objects/ba/51bc947259e5bb972a30d8d7d9b74d57df71e8 to s3://braindatatest/donuts/.git/objects/ba/51bc947259e5bb972a30d8d7d9b74d57df71e8\nCompleted 33 of 104 part(s) with 69 file(s) remaining\rupload: ../.git/objects/c6/5faa115321a2e36749fcfec56b8757e1219833 to s3://braindatatest/donuts/.git/objects/c6/5faa115321a2e36749fcfec56b8757e1219833\nCompleted 34 of 104 part(s) with 68 file(s) remaining\rupload: ../.git/objects/pack/pack-10c5fc35e8422b225308ed116d39d83a443b8c7d.idx to s3://braindatatest/donuts/.git/objects/pack/pack-10c5fc35e8422b225308ed116d39d83a443b8c7d.idx\nCompleted 35 of 104 part(s) with 67 file(s) remaining\rupload: ../.git/refs/heads/master to s3://braindatatest/donuts/.git/refs/heads/master\nCompleted 36 of 104 part(s) with 66 file(s) remaining\rupload: ../.git/refs/remotes/origin/HEAD to s3://braindatatest/donuts/.git/refs/remotes/origin/HEAD\nCompleted 37 of 104 part(s) with 65 file(s) remaining\rupload: ../.git/packed-refs to s3://braindatatest/donuts/.git/packed-refs\nCompleted 38 of 104 part(s) with 64 file(s) remaining\rupload: ../.git/refs/remotes/origin/master to s3://braindatatest/donuts/.git/refs/remotes/origin/master\nCompleted 39 of 104 part(s) with 63 file(s) remaining\rupload: ../.gitignore to s3://braindatatest/donuts/.gitignore\nCompleted 40 of 104 part(s) with 62 file(s) remaining\rupload: ../README.md to s3://braindatatest/donuts/README.md\nCompleted 41 of 104 part(s) with 61 file(s) remaining\rupload: ../.git/objects/f7/0e93d055d46969a0ceb03f7f79dd9e0942f864 to s3://braindatatest/donuts/.git/objects/f7/0e93d055d46969a0ceb03f7f79dd9e0942f864\nCompleted 42 of 104 part(s) with 60 file(s) remaining\rupload: ../TODO.md to s3://braindatatest/donuts/TODO.md\nCompleted 43 of 104 part(s) with 59 file(s) remaining\rupload: ../analyses/theoryComputingProx.ipynb to s3://braindatatest/donuts/analyses/theoryComputingProx.ipynb\nCompleted 44 of 104 part(s) with 58 file(s) remaining\rupload: ../analyses/miscSpark.ipynb to s3://braindatatest/donuts/analyses/miscSpark.ipynb\nCompleted 45 of 104 part(s) with 57 file(s) remaining\rupload: ../analyses/analysis_naive.ipynb to s3://braindatatest/donuts/analyses/analysis_naive.ipynb\nCompleted 46 of 104 part(s) with 56 file(s) remaining\rupload: ../analyses/tests.ipynb to s3://braindatatest/donuts/analyses/tests.ipynb\nCompleted 47 of 104 part(s) with 55 file(s) remaining\rupload: ../analyses/theoryConvSpline.ipynb to s3://braindatatest/donuts/analyses/theoryConvSpline.ipynb\nCompleted 48 of 104 part(s) with 54 file(s) remaining\rupload: ../analyses/theoryDeconvNcx.ipynb to s3://braindatatest/donuts/analyses/theoryDeconvNcx.ipynb\nCompleted 49 of 104 part(s) with 53 file(s) remaining\rupload: ../analyses/theoryEBP.ipynb to s3://braindatatest/donuts/analyses/theoryEBP.ipynb\nCompleted 50 of 104 part(s) with 52 file(s) remaining\rupload: ../analyses/theoryFitting.ipynb to s3://braindatatest/donuts/analyses/theoryFitting.ipynb\nCompleted 51 of 104 part(s) with 51 file(s) remaining\rupload: ../analyses/theoryGaussNewton.ipynb to s3://braindatatest/donuts/analyses/theoryGaussNewton.ipynb\nCompleted 52 of 104 part(s) with 50 file(s) remaining\rupload: ../analyses/theoryFitSph.ipynb to s3://braindatatest/donuts/analyses/theoryFitSph.ipynb\nCompleted 53 of 104 part(s) with 49 file(s) remaining\rCompleted 54 of 104 part(s) with 49 file(s) remaining\rupload: ../analyses/theoryGaussProx.ipynb to s3://braindatatest/donuts/analyses/theoryGaussProx.ipynb\nCompleted 55 of 104 part(s) with 48 file(s) remaining\rupload: ../analyses/theoryMLE.ipynb to s3://braindatatest/donuts/analyses/theoryMLE.ipynb\nCompleted 56 of 104 part(s) with 47 file(s) remaining\rupload: ../analyses/theoryScipyNcx2Contradiction.ipynb to s3://braindatatest/donuts/analyses/theoryScipyNcx2Contradiction.ipynb\nCompleted 57 of 104 part(s) with 46 file(s) remaining\rupload: ../doc/installation.rst to s3://braindatatest/donuts/doc/installation.rst\nCompleted 58 of 104 part(s) with 45 file(s) remaining\rupload: ../analyses/theorySpline.ipynb to s3://braindatatest/donuts/analyses/theorySpline.ipynb\nCompleted 59 of 104 part(s) with 44 file(s) remaining\rupload: ../analyses/theorySphHrm.ipynb to s3://braindatatest/donuts/analyses/theorySphHrm.ipynb\nCompleted 60 of 104 part(s) with 43 file(s) remaining\rupload: ../analyses/theory_misspecification.ipynb to s3://braindatatest/donuts/analyses/theory_misspecification.ipynb\nCompleted 61 of 104 part(s) with 42 file(s) remaining\rupload: ../doc/kappa.md to s3://braindatatest/donuts/doc/kappa.md\nCompleted 62 of 104 part(s) with 41 file(s) remaining\rupload: ../doc/latex/ncx2.tex to s3://braindatatest/donuts/doc/latex/ncx2.tex\nCompleted 63 of 104 part(s) with 40 file(s) remaining\rupload: ../doc/latex/writeup.tex to s3://braindatatest/donuts/doc/latex/writeup.tex\nCompleted 64 of 104 part(s) with 39 file(s) remaining\rupload: ../donuts/__init__.py to s3://braindatatest/donuts/donuts/__init__.py\nCompleted 65 of 104 part(s) with 38 file(s) remaining\rupload: ../donuts/data/100307bvals to s3://braindatatest/donuts/donuts/data/100307bvals\nCompleted 66 of 104 part(s) with 37 file(s) remaining\rupload: ../donuts/data/100307bvecs to s3://braindatatest/donuts/donuts/data/100307bvecs\nCompleted 67 of 104 part(s) with 36 file(s) remaining\rupload: ../donuts/data/data.py to s3://braindatatest/donuts/donuts/data/data.py\nCompleted 68 of 104 part(s) with 35 file(s) remaining\rupload: ../donuts/data/__init__.py to s3://braindatatest/donuts/donuts/data/__init__.py\nCompleted 69 of 104 part(s) with 34 file(s) remaining\rupload: ../donuts/data/roi1_b1000_1_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b1000_1_bvecs.csv\nCompleted 70 of 104 part(s) with 33 file(s) remaining\rupload: ../donuts/data/roi1_b1000_2_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b1000_2_bvecs.csv\nCompleted 71 of 104 part(s) with 32 file(s) remaining\rupload: ../donuts/data/100307_corpus_callosum.npy to s3://braindatatest/donuts/donuts/data/100307_corpus_callosum.npy\nCompleted 72 of 104 part(s) with 31 file(s) remaining\rupload: ../donuts/data/roi1_b2000_1_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b2000_1_bvecs.csv\nCompleted 73 of 104 part(s) with 30 file(s) remaining\rCompleted 74 of 104 part(s) with 30 file(s) remaining\rCompleted 75 of 104 part(s) with 30 file(s) remaining\rupload: ../donuts/data/roi1_b2000_2_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b2000_2_bvecs.csv\nCompleted 76 of 104 part(s) with 29 file(s) remaining\rupload: ../donuts/data/roi1_b4000_2_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b4000_2_bvecs.csv\nCompleted 77 of 104 part(s) with 28 file(s) remaining\rupload: ../donuts/data/100307small40thru80_100thru140_58.npy to s3://braindatatest/donuts/donuts/data/100307small40thru80_100thru140_58.npy\nCompleted 78 of 104 part(s) with 27 file(s) remaining\rupload: ../donuts/deconv/__init__.py to s3://braindatatest/donuts/donuts/deconv/__init__.py\nCompleted 79 of 104 part(s) with 26 file(s) remaining\rupload: ../donuts/data/roi1_b4000_1_bvecs.csv to s3://braindatatest/donuts/donuts/data/roi1_b4000_1_bvecs.csv\nCompleted 80 of 104 part(s) with 25 file(s) remaining\rupload: ../donuts/deconv/navigator.py to s3://braindatatest/donuts/donuts/deconv/navigator.py\nCompleted 81 of 104 part(s) with 24 file(s) remaining\rupload: ../.git/objects/pack/pack-10c5fc35e8422b225308ed116d39d83a443b8c7d.pack to s3://braindatatest/donuts/.git/objects/pack/pack-10c5fc35e8422b225308ed116d39d83a443b8c7d.pack\nCompleted 81 of 104 part(s) with 23 file(s) remaining\rupload: ../donuts/deconv/old_ncx.txt to s3://braindatatest/donuts/donuts/deconv/old_ncx.txt\nCompleted 82 of 104 part(s) with 22 file(s) remaining\rupload: ../donuts/deconv/ncx.py to s3://braindatatest/donuts/donuts/deconv/ncx.py\nCompleted 83 of 104 part(s) with 21 file(s) remaining\rupload: ../donuts/deconv/splines.py to s3://braindatatest/donuts/donuts/deconv/splines.py\nCompleted 84 of 104 part(s) with 20 file(s) remaining\rupload: ../donuts/emd/emd.h to s3://braindatatest/donuts/donuts/emd/emd.h\nCompleted 85 of 104 part(s) with 19 file(s) remaining\rupload: ../donuts/emd/pyemd.c to s3://braindatatest/donuts/donuts/emd/pyemd.c\nCompleted 86 of 104 part(s) with 18 file(s) remaining\rupload: ../donuts/data/roi1_b2000_1_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b2000_1_data.csv\nCompleted 87 of 104 part(s) with 17 file(s) remaining\rupload: ../donuts/emd/emd.c to s3://braindatatest/donuts/donuts/emd/emd.c\nCompleted 88 of 104 part(s) with 16 file(s) remaining\rupload: ../donuts/data/roi1_b2000_2_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b2000_2_data.csv\nCompleted 89 of 104 part(s) with 15 file(s) remaining\rupload: ../donuts/tests/test_deconv.py to s3://braindatatest/donuts/donuts/tests/test_deconv.py\nCompleted 90 of 104 part(s) with 14 file(s) remaining\rupload: ../donuts/tests/test_deconv_ncx.py to s3://braindatatest/donuts/donuts/tests/test_deconv_ncx.py\nCompleted 91 of 104 part(s) with 13 file(s) remaining\rupload: ../donuts/data/roi1_b4000_1_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b4000_1_data.csv\nCompleted 92 of 104 part(s) with 12 file(s) remaining\rupload: ../donuts/data/roi1_b1000_1_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b1000_1_data.csv\nCompleted 93 of 104 part(s) with 11 file(s) remaining\rupload: ../donuts/data/roi1_b1000_2_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b1000_2_data.csv\nCompleted 94 of 104 part(s) with 10 file(s) remaining\rupload: ../donuts/data/roi1_b4000_2_data.csv to s3://braindatatest/donuts/donuts/data/roi1_b4000_2_data.csv\nCompleted 95 of 104 part(s) with 9 file(s) remaining\rupload: ../setup.py to s3://braindatatest/donuts/setup.py\nCompleted 96 of 104 part(s) with 8 file(s) remaining\rupload: ../donuts/tests/test_deconv_utils.py to s3://braindatatest/donuts/donuts/tests/test_deconv_utils.py\nCompleted 97 of 104 part(s) with 7 file(s) remaining\rupload: ../donuts/tests/test_emd.py to s3://braindatatest/donuts/donuts/tests/test_emd.py\nCompleted 98 of 104 part(s) with 6 file(s) remaining\rupload: .ipynb_checkpoints/simulations-checkpoint.ipynb to s3://braindatatest/donuts/spark/.ipynb_checkpoints/simulations-checkpoint.ipynb\nCompleted 99 of 104 part(s) with 5 file(s) remaining\rupload: ./simulations.ipynb to s3://braindatatest/donuts/spark/simulations.ipynb\nCompleted 100 of 104 part(s) with 4 file(s) remaining\rupload: ./noise_est_single.ipynb to s3://braindatatest/donuts/spark/noise_est_single.ipynb\nCompleted 101 of 104 part(s) with 3 file(s) remaining\rupload: .ipynb_checkpoints/noise_est_single-checkpoint.ipynb to s3://braindatatest/donuts/spark/.ipynb_checkpoints/noise_est_single-checkpoint.ipynb\nCompleted 102 of 104 part(s) with 2 file(s) remaining\rupload: ../theory/gaussian_example.R to s3://braindatatest/donuts/theory/gaussian_example.R\nCompleted 103 of 104 part(s) with 1 file(s) remaining\rupload: ../donuts/deconv/utils.py to s3://braindatatest/donuts/donuts/deconv/utils.py\nupload: ../../computing/.git/COMMIT_EDITMSG to s3://braindatatest/computing/.git/COMMIT_EDITMSG\nCompleted 1 of 93 part(s) with 90 file(s) remaining\rupload: ../../computing/.git/ORIG_HEAD to s3://braindatatest/computing/.git/ORIG_HEAD\nCompleted 2 of 93 part(s) with 89 file(s) remaining\rupload: ../../computing/.git/HEAD to s3://braindatatest/computing/.git/HEAD\nCompleted 3 of 93 part(s) with 88 file(s) remaining\rupload: ../../computing/.git/hooks/applypatch-msg.sample to s3://braindatatest/computing/.git/hooks/applypatch-msg.sample\nCompleted 4 of 93 part(s) with 87 file(s) remaining\rupload: ../../computing/.git/description to s3://braindatatest/computing/.git/description\nCompleted 5 of 93 part(s) with 86 file(s) remaining\rupload: ../../computing/.git/FETCH_HEAD to s3://braindatatest/computing/.git/FETCH_HEAD\nCompleted 6 of 93 part(s) with 85 file(s) remaining\rupload: ../../computing/.git/hooks/pre-applypatch.sample to s3://braindatatest/computing/.git/hooks/pre-applypatch.sample\nCompleted 7 of 93 part(s) with 84 file(s) remaining\rupload: ../../computing/.git/hooks/commit-msg.sample to s3://braindatatest/computing/.git/hooks/commit-msg.sample\nCompleted 8 of 93 part(s) with 83 file(s) remaining\rupload: ../../computing/.git/hooks/pre-push.sample to s3://braindatatest/computing/.git/hooks/pre-push.sample\nCompleted 9 of 93 part(s) with 82 file(s) remaining\rupload: ../../computing/.git/hooks/pre-commit.sample to s3://braindatatest/computing/.git/hooks/pre-commit.sample\nCompleted 10 of 93 part(s) with 81 file(s) remaining\rupload: ../../computing/.git/hooks/post-update.sample to s3://braindatatest/computing/.git/hooks/post-update.sample\nCompleted 11 of 93 part(s) with 80 file(s) remaining\rupload: ../../computing/.git/config to s3://braindatatest/computing/.git/config\nCompleted 12 of 93 part(s) with 79 file(s) remaining\rupload: ../../computing/.git/hooks/prepare-commit-msg.sample to s3://braindatatest/computing/.git/hooks/prepare-commit-msg.sample\nCompleted 13 of 93 part(s) with 78 file(s) remaining\rupload: ../../computing/.git/hooks/pre-rebase.sample to s3://braindatatest/computing/.git/hooks/pre-rebase.sample\nCompleted 14 of 93 part(s) with 77 file(s) remaining\rupload: ../../computing/.git/hooks/update.sample to s3://braindatatest/computing/.git/hooks/update.sample\nCompleted 15 of 93 part(s) with 76 file(s) remaining\rupload: ../../computing/.git/index to s3://braindatatest/computing/.git/index\nCompleted 16 of 93 part(s) with 75 file(s) remaining\rupload: ../../computing/.git/info/exclude to s3://braindatatest/computing/.git/info/exclude\nCompleted 17 of 93 part(s) with 74 file(s) remaining\rupload: ../../computing/.git/logs/HEAD to s3://braindatatest/computing/.git/logs/HEAD\nCompleted 18 of 93 part(s) with 73 file(s) remaining\rupload: ../../computing/.git/logs/refs/heads/master to s3://braindatatest/computing/.git/logs/refs/heads/master\nCompleted 19 of 93 part(s) with 72 file(s) remaining\rupload: ../../computing/.git/logs/refs/remotes/origin/HEAD to s3://braindatatest/computing/.git/logs/refs/remotes/origin/HEAD\nCompleted 20 of 93 part(s) with 71 file(s) remaining\rupload: ../../computing/.git/logs/refs/remotes/origin/master to s3://braindatatest/computing/.git/logs/refs/remotes/origin/master\nCompleted 21 of 93 part(s) with 70 file(s) remaining\rupload: ../../computing/.git/objects/11/7e5e2ceb8df61646d1d088c408deacfd4d3f43 to s3://braindatatest/computing/.git/objects/11/7e5e2ceb8df61646d1d088c408deacfd4d3f43\nCompleted 22 of 93 part(s) with 69 file(s) remaining\rupload: ../../computing/.git/objects/0f/0905aac3877916aa5d010b635c9c36111c6196 to s3://braindatatest/computing/.git/objects/0f/0905aac3877916aa5d010b635c9c36111c6196\nCompleted 23 of 93 part(s) with 68 file(s) remaining\rupload: ../../computing/.git/objects/00/1eed3045fa482290cb1febd58bfc66d8a76499 to s3://braindatatest/computing/.git/objects/00/1eed3045fa482290cb1febd58bfc66d8a76499\nCompleted 24 of 93 part(s) with 67 file(s) remaining\rupload: ../../computing/.git/objects/1a/b0489b81c23daf9dd3dd470fe3583abcd252a0 to s3://braindatatest/computing/.git/objects/1a/b0489b81c23daf9dd3dd470fe3583abcd252a0\nCompleted 25 of 93 part(s) with 66 file(s) remaining\rupload: ../../computing/.git/objects/29/5613ba62198502c4f1b5204e2a3649dd5a8feb to s3://braindatatest/computing/.git/objects/29/5613ba62198502c4f1b5204e2a3649dd5a8feb\nCompleted 26 of 93 part(s) with 65 file(s) remaining\rupload: ../../computing/.git/objects/3f/befc259d45236aba7495688c6c85fcf7afbcd1 to s3://braindatatest/computing/.git/objects/3f/befc259d45236aba7495688c6c85fcf7afbcd1\nCompleted 27 of 93 part(s) with 64 file(s) remaining\rupload: ../../computing/.git/objects/1c/3890676a5008dbdc6aabd17c51c0ff597126e4 to s3://braindatatest/computing/.git/objects/1c/3890676a5008dbdc6aabd17c51c0ff597126e4\nCompleted 28 of 93 part(s) with 63 file(s) remaining\rupload: ../../computing/.git/objects/21/93fb7b53a43b9a6abd5bfee7e7b2fbfd178c01 to s3://braindatatest/computing/.git/objects/21/93fb7b53a43b9a6abd5bfee7e7b2fbfd178c01\nCompleted 29 of 93 part(s) with 62 file(s) remaining\rupload: ../../computing/.git/objects/45/d9e23e7aaf27389dda83a4ddd4e37dcb749c6e to s3://braindatatest/computing/.git/objects/45/d9e23e7aaf27389dda83a4ddd4e37dcb749c6e\nCompleted 30 of 93 part(s) with 61 file(s) remaining\rupload: ../../computing/.git/objects/4a/059e139a05e1c2c66bbae7dcc461b3bff61e24 to s3://braindatatest/computing/.git/objects/4a/059e139a05e1c2c66bbae7dcc461b3bff61e24\nCompleted 31 of 93 part(s) with 60 file(s) remaining\rupload: ../../computing/.git/objects/51/4b8ab459d2094d9e2bf8ce03e99a4d68c04105 to s3://braindatatest/computing/.git/objects/51/4b8ab459d2094d9e2bf8ce03e99a4d68c04105\nCompleted 32 of 93 part(s) with 59 file(s) remaining\rupload: ../../computing/.git/objects/4b/51b2c78602920fcc8248ab215d8f5c1a46635c to s3://braindatatest/computing/.git/objects/4b/51b2c78602920fcc8248ab215d8f5c1a46635c\nCompleted 33 of 93 part(s) with 58 file(s) remaining\rupload: ../../computing/.git/objects/6b/56f725f11c7c525ae09ea7a86a0f704b1e7040 to s3://braindatatest/computing/.git/objects/6b/56f725f11c7c525ae09ea7a86a0f704b1e7040\nCompleted 34 of 93 part(s) with 57 file(s) remaining\rupload: ../../computing/.git/objects/70/434c842b3500e39dd1144b6fa7530d08c96dbb to s3://braindatatest/computing/.git/objects/70/434c842b3500e39dd1144b6fa7530d08c96dbb\nCompleted 35 of 93 part(s) with 56 file(s) remaining\rupload: ../../computing/.git/objects/7b/bd0685a00eb6294cbe75804579e34b30f48fa0 to s3://braindatatest/computing/.git/objects/7b/bd0685a00eb6294cbe75804579e34b30f48fa0\nCompleted 36 of 93 part(s) with 55 file(s) remaining\rupload: ../../computing/.git/objects/8f/b89e530f255c45954866ed604e8ed036af3df1 to s3://braindatatest/computing/.git/objects/8f/b89e530f255c45954866ed604e8ed036af3df1\nCompleted 37 of 93 part(s) with 54 file(s) remaining\rupload: ../../computing/.git/objects/83/c4a7782a62b323bd9a711454cd623fa182cf8e to s3://braindatatest/computing/.git/objects/83/c4a7782a62b323bd9a711454cd623fa182cf8e\nCompleted 38 of 93 part(s) with 53 file(s) remaining\rupload: ../../computing/.git/objects/91/0bab45bb400b96cdadec2a8dfd9bd1c4141b99 to s3://braindatatest/computing/.git/objects/91/0bab45bb400b96cdadec2a8dfd9bd1c4141b99\nCompleted 39 of 93 part(s) with 52 file(s) remaining\rupload: ../../computing/.git/objects/94/ffea95caf6da2b8650271cfa4ce2d27c7e863f to s3://braindatatest/computing/.git/objects/94/ffea95caf6da2b8650271cfa4ce2d27c7e863f\nCompleted 40 of 93 part(s) with 51 file(s) remaining\rupload: ../../computing/.git/objects/97/83a4a092b206d1aa1f678749e495b57571d94c to s3://braindatatest/computing/.git/objects/97/83a4a092b206d1aa1f678749e495b57571d94c\nCompleted 41 of 93 part(s) with 50 file(s) remaining\rupload: ../../computing/.git/objects/9f/74f883e4ed16c9fcba0dcd6b08cb544371cbd5 to s3://braindatatest/computing/.git/objects/9f/74f883e4ed16c9fcba0dcd6b08cb544371cbd5\nCompleted 42 of 93 part(s) with 49 file(s) remaining\rupload: ../../computing/.git/objects/9e/b84cfc56464aaf98ea91f08065e39e516255fa to s3://braindatatest/computing/.git/objects/9e/b84cfc56464aaf98ea91f08065e39e516255fa\nCompleted 43 of 93 part(s) with 48 file(s) remaining\rupload: ../../computing/.git/objects/a1/92a00ea4a6c2279e2bded0f5d5b0e995b3f26d to s3://braindatatest/computing/.git/objects/a1/92a00ea4a6c2279e2bded0f5d5b0e995b3f26d\nCompleted 44 of 93 part(s) with 47 file(s) remaining\rupload: ../../computing/.git/objects/b1/1d55f252564d0a83c1b740f26bd11e5c966911 to s3://braindatatest/computing/.git/objects/b1/1d55f252564d0a83c1b740f26bd11e5c966911\nCompleted 45 of 93 part(s) with 46 file(s) remaining\rupload: ../../computing/.git/objects/ab/97a97966ed9343a2c16332775dce2058b266e6 to s3://braindatatest/computing/.git/objects/ab/97a97966ed9343a2c16332775dce2058b266e6\nCompleted 46 of 93 part(s) with 45 file(s) remaining\rupload: ../../computing/.git/objects/be/3ed2faf395a4660ca4cf59d5d75115166767dd to s3://braindatatest/computing/.git/objects/be/3ed2faf395a4660ca4cf59d5d75115166767dd\nCompleted 47 of 93 part(s) with 44 file(s) remaining\rupload: ../../computing/.git/objects/bc/c7996d637124011c8114eaa0564158310a7343 to s3://braindatatest/computing/.git/objects/bc/c7996d637124011c8114eaa0564158310a7343\nCompleted 48 of 93 part(s) with 43 file(s) remaining\rupload: ../../computing/.git/objects/c3/8ba32418df9cb6e97d8cbd4d99ca78040b21f3 to s3://braindatatest/computing/.git/objects/c3/8ba32418df9cb6e97d8cbd4d99ca78040b21f3\nCompleted 49 of 93 part(s) with 42 file(s) remaining\rupload: ../../computing/.git/objects/d4/f0d952564d10620faf9500300c8e3bc409429f to s3://braindatatest/computing/.git/objects/d4/f0d952564d10620faf9500300c8e3bc409429f\nCompleted 50 of 93 part(s) with 41 file(s) remaining\rupload: ../../computing/.git/objects/d6/c0baaa115a5f6385637123f9150a557c68fe4b to s3://braindatatest/computing/.git/objects/d6/c0baaa115a5f6385637123f9150a557c68fe4b\nCompleted 51 of 93 part(s) with 40 file(s) remaining\rupload: ../../computing/.git/objects/ec/8300baa89137bb3c1c597648ffffa44ca0fe59 to s3://braindatatest/computing/.git/objects/ec/8300baa89137bb3c1c597648ffffa44ca0fe59\nCompleted 52 of 93 part(s) with 39 file(s) remaining\rupload: ../../computing/.git/objects/d7/f186e4ca407276bbae9fedf7145ca82a4a8982 to s3://braindatatest/computing/.git/objects/d7/f186e4ca407276bbae9fedf7145ca82a4a8982\nCompleted 53 of 93 part(s) with 38 file(s) remaining\rupload: ../../computing/.git/objects/fd/4f4d7c4403feee0842a822f02b4dbd27a80624 to s3://braindatatest/computing/.git/objects/fd/4f4d7c4403feee0842a822f02b4dbd27a80624\nCompleted 54 of 93 part(s) with 37 file(s) remaining\rupload: ../../computing/.git/objects/fc/2bdf5aae56bf7297f843b31fa12c96a7445e71 to s3://braindatatest/computing/.git/objects/fc/2bdf5aae56bf7297f843b31fa12c96a7445e71\nCompleted 55 of 93 part(s) with 36 file(s) remaining\rupload: ../../computing/.git/objects/ff/cf00b01fa7ed4613afeb5c21da2a3afbc7290f to s3://braindatatest/computing/.git/objects/ff/cf00b01fa7ed4613afeb5c21da2a3afbc7290f\nCompleted 56 of 93 part(s) with 35 file(s) remaining\rupload: ../../computing/.git/objects/pack/pack-f1f5a0c08b617bede94846c87ba6cb05373176ae.idx to s3://braindatatest/computing/.git/objects/pack/pack-f1f5a0c08b617bede94846c87ba6cb05373176ae.idx\nCompleted 57 of 93 part(s) with 34 file(s) remaining\rupload: ../../computing/.git/packed-refs to s3://braindatatest/computing/.git/packed-refs\nCompleted 58 of 93 part(s) with 33 file(s) remaining\rupload: ../../computing/.git/refs/heads/master to s3://braindatatest/computing/.git/refs/heads/master\nCompleted 59 of 93 part(s) with 32 file(s) remaining\rupload: ../../computing/.git/refs/remotes/origin/HEAD to s3://braindatatest/computing/.git/refs/remotes/origin/HEAD\nCompleted 60 of 93 part(s) with 31 file(s) remaining\rupload: ../../computing/.git/refs/remotes/origin/master to s3://braindatatest/computing/.git/refs/remotes/origin/master\nCompleted 61 of 93 part(s) with 30 file(s) remaining\rupload: ../../computing/bototest1.ipynb to s3://braindatatest/computing/bototest1.ipynb\nCompleted 62 of 93 part(s) with 29 file(s) remaining\rupload: ../../computing/bototest2.ipynb to s3://braindatatest/computing/bototest2.ipynb\nCompleted 63 of 93 part(s) with 28 file(s) remaining\rupload: ../../computing/examples/README.md to s3://braindatatest/computing/examples/README.md\nCompleted 64 of 93 part(s) with 27 file(s) remaining\rupload: ../../computing/examples/install.sh to s3://braindatatest/computing/examples/install.sh\nCompleted 65 of 93 part(s) with 26 file(s) remaining\rupload: ../../computing/examples/.ipynb_checkpoints/test2-checkpoint.ipynb to s3://braindatatest/computing/examples/.ipynb_checkpoints/test2-checkpoint.ipynb\nCompleted 66 of 93 part(s) with 25 file(s) remaining\rupload: ../../computing/examples/included_examples.ipynb to s3://braindatatest/computing/examples/included_examples.ipynb\nCompleted 67 of 93 part(s) with 24 file(s) remaining\rupload: ../../computing/README.md to s3://braindatatest/computing/README.md\nCompleted 68 of 93 part(s) with 23 file(s) remaining\rupload: ../../computing/examples/restart.sh to s3://braindatatest/computing/examples/restart.sh\nCompleted 69 of 93 part(s) with 22 file(s) remaining\rupload: ../../computing/notes.txt to s3://braindatatest/computing/notes.txt\nCompleted 70 of 93 part(s) with 21 file(s) remaining\rupload: ../../computing/master_rules.ods to s3://braindatatest/computing/master_rules.ods\nCompleted 71 of 93 part(s) with 20 file(s) remaining\rupload: ../../computing/slave_rules.ods to s3://braindatatest/computing/slave_rules.ods\nCompleted 72 of 93 part(s) with 19 file(s) remaining\rupload: ../../computing/examples/test2.ipynb to s3://braindatatest/computing/examples/test2.ipynb\nCompleted 73 of 93 part(s) with 18 file(s) remaining\rupload: ../../computing/spark_test.ipynb to s3://braindatatest/computing/spark_test.ipynb\nCompleted 74 of 93 part(s) with 17 file(s) remaining\rupload: ../../computing/study_these/config.txt to s3://braindatatest/computing/study_these/config.txt\nCompleted 75 of 93 part(s) with 16 file(s) remaining\rupload: ../../computing/tutorial/README.md to s3://braindatatest/computing/tutorial/README.md\nCompleted 76 of 93 part(s) with 15 file(s) remaining\rupload: ../../computing/study_these/spark_ec2.py to s3://braindatatest/computing/study_these/spark_ec2.py\nCompleted 77 of 93 part(s) with 14 file(s) remaining\rCompleted 78 of 93 part(s) with 14 file(s) remaining\rupload: ../../computing/tutorial/assets/ami3.png to s3://braindatatest/computing/tutorial/assets/ami3.png\nCompleted 79 of 93 part(s) with 13 file(s) remaining\rupload: ../../computing/tutorial/assets/ami2.png to s3://braindatatest/computing/tutorial/assets/ami2.png\nCompleted 80 of 93 part(s) with 12 file(s) remaining\rupload: ../../computing/tutorial/assets/ami4.png to s3://braindatatest/computing/tutorial/assets/ami4.png\nCompleted 81 of 93 part(s) with 11 file(s) remaining\rupload: ../../computing/tutorial/assets/ami1.png to s3://braindatatest/computing/tutorial/assets/ami1.png\nCompleted 82 of 93 part(s) with 10 file(s) remaining\rupload: ../../computing/tutorial/assets/ami5.png to s3://braindatatest/computing/tutorial/assets/ami5.png\nCompleted 83 of 93 part(s) with 9 file(s) remaining\rCompleted 84 of 93 part(s) with 9 file(s) remaining\rupload: ../../computing/tutorial/assets/keypair.png to s3://braindatatest/computing/tutorial/assets/keypair.png\nCompleted 85 of 93 part(s) with 8 file(s) remaining\rupload: ../../computing/tutorial/assets/security_edit.png to s3://braindatatest/computing/tutorial/assets/security_edit.png\nCompleted 86 of 93 part(s) with 7 file(s) remaining\rupload: ../../computing/tutorial/part1_home.md to s3://braindatatest/computing/tutorial/part1_home.md\nCompleted 87 of 93 part(s) with 6 file(s) remaining\rupload: ../../computing/tutorial/assets/security_edit2.png to s3://braindatatest/computing/tutorial/assets/security_edit2.png\nCompleted 88 of 93 part(s) with 5 file(s) remaining\rCompleted 89 of 93 part(s) with 5 file(s) remaining\rupload: ../../computing/tutorial/part2_ec2_python.md to s3://braindatatest/computing/tutorial/part2_ec2_python.md\nCompleted 90 of 93 part(s) with 4 file(s) remaining\rupload: ../../computing/tutorial/part3_ec2_ubuntu_sparkR.md to s3://braindatatest/computing/tutorial/part3_ec2_ubuntu_sparkR.md\nCompleted 91 of 93 part(s) with 3 file(s) remaining\rupload: ../../computing/tutorial/assets/publicDNS.png to s3://braindatatest/computing/tutorial/assets/publicDNS.png\nCompleted 92 of 93 part(s) with 2 file(s) remaining\rupload: ../../computing/.git/objects/pack/pack-f1f5a0c08b617bede94846c87ba6cb05373176ae.pack to s3://braindatatest/computing/.git/objects/pack/pack-f1f5a0c08b617bede94846c87ba6cb05373176ae.pack\nCompleted 92 of 93 part(s) with 1 file(s) remaining\rCompleted 93 of 93 part(s) with 1 file(s) remaining\rupload: ../../computing/spark-1.1.0.tgz to s3://braindatatest/computing/spark-1.1.0.tgz\n'




    subprocess.check_output('git -C /root/donuts commit -a -m "tested EMD in simulations.ipynb"', shell=True)




    '[master dd00708] tested EMD in simulations.ipynb\n Committer: Charles EC2 <root@ip-172-31-20-178.us-west-2.compute.internal>\nYour name and email address were configured automatically based\non your username and hostname. Please check that they are accurate.\nYou can suppress this message by setting them explicitly:\n\n    git config --global user.name "Your Name"\n    git config --global user.email you@example.com\n\nAfter doing this, you may fix the identity used for this commit with:\n\n    git commit --amend --reset-author\n\n 1 file changed, 97 insertions(+), 39 deletions(-)\n'




    