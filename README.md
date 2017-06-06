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
git clone https://github.com/mmongia/SSLearningPipeLine.git
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


# How to get error results

We first need to edit user_driver_m.py file. In the main function where the following lines of code are written 
```
    for idx in A:
        #break
        #locdata = get_old_info(xml)

```
edit it so that reads. 
```
    for idx in A:
        break
        #locdata = get_old_info(xml)
```

Run the code using
```
PYTHONPATH=../pylabelme:$PYTHONPATH python user_driver_m
```
and from code located in sslearningpipeline.py, a graph plotting the errors of the predicted boxes will be produced.
The graph should look similar to  below.



![alt text](https://github.com/mmongia/SSLearningPipeLine/blob/master/ErrorFromTransferLearning.JPG)





# How to label images




