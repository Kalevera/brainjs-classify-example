# brainjs-classify-example

This module is meant to show an example for classifying string data, for use in sentiment analysis scenarios and survey procecssing. Particularly it is meant to classify types of sentences into questions or declerations.

## Brief aside about how this process flows to provide the output and why this projcet works.
The projcet leverages Supervised FFN

The layers beeing generated are a Feed Forward Neural Network and pulling in text data to a nueral network is is accomplished by reading individual chracters as data "(a),(b),(c),(etc.)" in an array and converting it to an ASCII numeric value.

In 2012 prior to auto encoders many data operations were manual. 
So taking the converted characters the data opertaions pass over the vector (input array) and divide by 255 (ASCII character range), this "normalizes" the data set to make it easier for the machines to ingest. This was crucial back then when tools were slim to none.

In machine learning inputs cannot varry. So you must select the input vector shape and format. (3D array w,k,m) or in this case for sentences a single dimensional array (1D array n-length), for this experiment.

Then adjust the inputs further as they go into the networks by padding both the inputs and outputs to match the maxium input array length and maximum output array length (this is because the arrays varry significantly enough to cause errors in the training) there for we fixed the length. 

In this case I solved this with a query to the data-set and found the largest array within and using that argmax search to set the maxarray length variable. I then shifted everything into a vector starting at position 0 and filling the array with the input data and then padding (zeros) where some sentences where shorter than the max sentence length.

Once the padding (zeros) is added the softwares flow moves stream data through the Brain.js Nueral Network. (using node.js to file stream the data) ```--side note why even use node? well GPUs were not the main stay most libraries only worked on the cpu. therefore even though the single threaded nature of nodejs wasn't good for this task it was still able to process the software demand and train the network with suprising accuracy. ```

Back then you would then Wait a few days while the error is reduced.
Once complete test the network comparability using the teacher.answer() method - this approach of trying to get the network to spell was testing to see if networks could "speak" back at this rudementary level. They could not but what is a side affect of this exploration is that the model is able to determine if a question is asked of and showing that they could be flexible enough to teach different abstract recognitions, this technique can be used to pull social media or user responses and determine a specific sentiment (negative, positive, neutral, etc.). 

With these networks it gave rise to the early abilities of neural networks being used in Natural Language Processing and particularly sentiment analysis.

This would eventually be used for anomoly detection. Plenty of signals had good quality outputs it was when the AI scored low e.g. it did not make a correct match that you knew the AI detected any anomoly.

### An example interacting with the FFN (AI) is using the teacher module.

```
let brainTeacher = require ('./index-stream.js);
let teacher = new brainTeacher({source:'/data/numericsentence.JSON'});

teacher.loadDatasource(); //this will begin reading the data source provided and automatically begin training
```
the file specified in this example will take days to process given the example setup (CPU based)

### Testing out the nueral network once training has completed.
there should already be a test string in the example but if you wish to look at more about  
the understaing that the network obtained you can test more using the below method.
~~~~
teacher.answer("What is your name?")
~~~~

### screen shots of example run

![](https://github.com/Kalevera/brainjs-classify-example/blob/master/img/kit_train_info.png?raw=true|alt=training_information)

The above shows the results of the current setup in index-stream.js.
You can see that it's getting close to identifying "question" ("qtessiom", "ptershnn") and from the data it learned ("my") and ("name") strings note that it is even getting close to classifying it self as "you". In the context of the training data the FFN is represented as "kit" and "my" should return a classification of "my".
You can see the Network struggles in other areas in below test strings.

![](https://github.com/Kalevera/brainjs-classify-example/blob/master/img/kit_train_answers.png?raw=true|alt=training_answers)

The above shows results of the network being tested. Words like "who, what, when, where, and why" whould return a classification of question based on the output data supplied to the network. We can see that it struggles a little bit to actually spell the words but its providing an underlying representation with "where".  Because the Network was trained to return "location" on those output data points. 

### Key Output training notes
Classifications were attempted in the following manner.
1) If the sentence had a question word it would be considered a question. All other sentences would be a decleration.
2) If the sentence contained you, your, you're it would be considered "kit" the FFN in this example, just felt fun to name it at this point in time.
3) Sentences with "me, my, time" the networked was trained to return a refernce of that word to determine if the sentence may be requesting a timer or alarm. 
