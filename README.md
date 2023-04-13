[![Documentation Status](https://readthedocs.org/projects/sydr/badge/?version=latest)](https://sydr.readthedocs.io/en/latest/?badge=latest)

# SyDR

SyDR is an open-source Software Defined Radio (SDR) for GNSS processing developed in Python. SyDR's goal is to provide a controlled environment for testing new processing algorithms, for benchmarking purposes.

The software is still at the very early development stages, with limited testing of the software functionalities. Issues and bugs are to be expected with the current version. 

# Requirements

The software has been developed and tested on Python 3.11 on a Windows Linux Subsytem (WSL) version 2. 
It is recommand to create a python virtual environement after cloning the directory. 

`python3 -m venv env`

`source env/bin/activate`

To install the different libraries required, use the `pip` command line with the `requirements.txt`. 

`pip install -r requirements.txt`

To create a directory hosting the output results
`mkdir .results`

# OS support

The design had been tested in WSL2 as well as x86 Ubuntu 20.04.5 LTS
### Mac User
This design may required modification in order to run in MacOS and some Unix system. This is due to `._get_value()` method of the semaphore in Python's multiprocessing module on some unix system has not been implemented. You will come across with error such as:
```bash
return self._maxsize - self._sem._semlock._get_value()
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NotImplementedError
```

# Documentation 

TBD
