Jack wrote this for me so I can set it up again

# To create the local virtualenv
C:\Users\oaw001\AppData\Local\Programs\Python\Python312\python.exe -m pip install virtualenv
C:\Users\oaw001\AppData\Local\Programs\Python\Python312\python.exe -m virtualenv .venv 

## When you start a powershell instance (must be in this directory)
.\.venv\Scripts\activate.bat

## You can use python line by line
python 
>> 1+1

## Or call a file to run
python MainScript.py