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

![](https://github.com/Kalevera/brainjs-classify-example/blob/master/img/kit_train_info.png?raw=true|alt=training_information)

The above shows the results of the current setup in index-stream.js.
You can see that it's getting close to identifying "question" ("qtessiom", "ptershnn my") strings note that it is even getting close to classifying it self as you in the context of the training data is represented as "kit" and "my" should return a classification of "my".
You can see the Network struggles in other areas in below test strings.

![](https://github.com/Kalevera/brainjs-classify-example/blob/master/img/kit_train_answers.png?raw=true|alt=training_answers)

The above shows results of the network being tested. Words like "who, what, when, where, and why" whould return a classification of question based on the output data supplied to the network. We can see that it struggles a little bit with "where".  Because the Network was trained to return "location" on those output data points. 

###Key Output training notes
Classifications were attempted in the following manner.
1) If the sentence had a question word it would be considered a question. All other sentences would be a decleration.
2) If the sentence contained you, your, you're it would be considered "kit". As that is what i named my particular network.
3) Sentences with "me, my, time" than it should also return a refernce of that word to determine if the sentence may be requesting a timer or alarm. 
