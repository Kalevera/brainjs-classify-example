# brainjs-classify-example

This module is meant to show an example for classifying string data. Particularly it is meant to classify types of sentences
into questions or declerations.

###Technique Used
The technique used is accomplished by read the data and converting it to an ASCII numeric value.
Next divide by 255 (ASCII character range).
Then pad both the inputs and outputs to match the maxium input array length and maximum output array length (this is because the arrays varry significantly enough to cause errors in the training)
Once the padding (zeros) is added I then run the data through the Brain.js Nueral Network.
Wait a few days while the error is reduced.
Once complete test the network comprability using the teacher.answer() method

###An example of the teacher module would be. 

~~~~
let brainTeacher = require ('./index-stream.js);
let teacher = new brainTeacher({source:'/data/numericsentence.JSON'});

teacher.loadDatasource(); //this will begin reading the data source provided and automatically begin training
 
~~~~
the file specified in this example will take days to process given the example setup

### Testing out the nueral network once training has completed.
there should already be a test string in the example but if you wish to look at more stringuage 
the understaing that the network obtained you can test more using the below method.
~~~~
teacher.answer("What is your name?")
~~~~

###screen shots of example run
