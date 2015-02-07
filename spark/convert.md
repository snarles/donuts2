

    import os
    import subprocess


    files = os.listdir('.')
    ipynbs = [file for file in files if '.ipynb' in file and '_checkpoints' not in file]
    for ipynb in ipynbs:
        subprocess.check_output('ipython nbconvert --to html '+ipynb, shell=True)
    subprocess.check_output('git add *.ipynb *.html', shell=True)
    subprocess.check_output('git commit -a -m "Auto HTML Conversion"', shell=True)




    '[master da1a63c] Auto HTML Conversion\n 11 files changed, 34811 insertions(+)\n create mode 100644 spark/Untitled0.html\n create mode 100644 spark/chris1.html\n create mode 100644 spark/chris2_exploratory.html\n create mode 100644 spark/chrisSC.html\n create mode 100644 spark/convert.html\n create mode 100644 spark/convert.ipynb\n create mode 100644 spark/noise_est_single.html\n create mode 100644 spark/proc_data.html\n create mode 100644 spark/proc_data_b.html\n create mode 100644 spark/simulations.html\n create mode 100644 spark/sparkDonuts.html\n'




    
