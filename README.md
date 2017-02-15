# SSLearningPipeLine

intro

# Operations

You need to have both this repository, and davidslac/pylabelme checked out. Then before you run, you need to adjust 
your PYTHONPATH so that when you run the SSLearningPipeline user_driver.py, it can find the labelme tool.

Suggestion,

create a working directory, ie

```
mkdir work
cd work
```

source conda_setup
on pslogin (outside internet machine, get both these repos):

```
git clone https://github.com/davidslac/SSLearningPipeLine.git
git clone https://github.com/davidslac/pylabelme.git
```

now in another terminal, 

```
ssh psana
source conda_setup
cd work/SSLearningPipeline

PYTHONPATH=../pylabelme:$PYTHONPATH python user_driver.py
```

notice that the script, user_driver.py, is telling sslearn to write the labeled files into 
```
/reg/d/psdm/amo/amo86815/scratch/davidsch
```
to get going, make your own directory in scratch, edit user_driver.py for yourself.

