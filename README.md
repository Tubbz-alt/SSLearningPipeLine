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
Make sure in the main function in user_driver_m.py that the code reads like the following


```
    for idx in A:
        #break
        #locdata = get_old_info(xml)

```

Now run the code using the following 

```
PYTHONPATH=../pylabelme:$PYTHONPATH python user_driver_m
```
You will see that many images come up one after another. These images are already labeled. Eventually there will come an image that has not been labeled and you will have a choice to label it or not. You can click enter on the command promp to label. You can type "n" and then click enter to move on. Make sure to label images that only have fingers on the top island. For the sake of this tutorial we have set up the code to worry about the top finger. 

![alt text](https://github.com/mmongia/SSLearningPipeLine/blob/master/Comment1.JPG)
![alt text](https://github.com/mmongia/SSLearningPipeLine/blob/master/Comment2.JPG)
![alt text](https://github.com/mmongia/SSLearningPipeLine/blob/master/comment3.JPG)



If you do decide to label. 




