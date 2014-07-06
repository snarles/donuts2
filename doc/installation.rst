## Installing `donuts` for development

In the top-level directory of the repo, run:

    python setup.py install --prefix=~/path/to/a/directory

To be able to import donuts, be sure to add the following to your ~/.bashrc file: 

    export PYTHONPATH=$PYTHONPATH:/home/username/path/to/a/directory/lib/python2.7/site-packages/

If you want to avoid having to run the python installation step every time you make changes to the code, you can also symlink from your python path into your source directory, for example: 

    ln -s ~/projects/donuts/donuts/ ~/usr/lib/python2.7/site-packages/donuts 

Where ~/usr/lib/python2.7/site-packages/ is a location added to your environment PYTHONPATH variable. 
    
## Running the tests

With the top-level directory of the repo type:

    nosetests


