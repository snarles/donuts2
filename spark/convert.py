
# coding: utf-8

# In[8]:

import os
import subprocess


# In[15]:

files = os.listdir('.')
ipynbs = [file for file in files if '.ipynb' in file and '_checkpoints' not in file]
for ipynb in ipynbs:
    subprocess.check_output('ipython nbconvert --to html '+ipynb, shell=True)
subprocess.check_output('git add *.ipynb *.html', shell=True)
subprocess.check_output('git commit -a -m "Auto HTML Conversion"', shell=True)


# In[ ]:


