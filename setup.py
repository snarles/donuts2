#!/usr/bin/env python
from distutils.core import setup, Extension


EXTENSIONS = [dict(name="donuts.emd",
                   sources=["donuts/emd/pyemd.c", "donuts/emd/emd.c"],
                   extra_compile_args=['-g'])]

opts = dict(name='donuts',
            packages=['donuts',
                      'donuts.emd',
                      'donuts.deconv'],
            ext_modules = [Extension(**e) for e in EXTENSIONS])

if __name__ == '__main__':
    setup(**opts)
