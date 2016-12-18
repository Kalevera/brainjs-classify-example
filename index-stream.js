//index-stream.js
var brain = require('./brain.js/dist/brain'),
	fs = require('fs');

var BrainTeacher = function (opts){
  if(!opts) throw new Error('data source file not specified');
  this.dataSource = opts.source || null;
  this.orgDataSet = null;
  this.dataSet = [];
  this.tempDataSet = null;
  this.maxInputLength= null;
  this.maxOutputLength = null;
  this.pastTests = {inputs:[null],correct:[null]};
  this.trainedNet = null;
  this.nueral = null;
}

BrainTeacher.prototype ={
  loadDatasource: function (){
    let self = this;
    let streamer = fs.createReadStream(__dirname+this.dataSource);
    let tempobj;
    streamer.pause();
    streamer.on('readable',function(){
      let chunk;
      while(chunk = streamer.read()){
        self.orgDataSet = JSON.parse(chunk);
      }
    });
    streamer.on('end',function(){
      console.log('data done reading');

      self.setupData();
    })
  },
  setupData: function (){
    
    let temp = this.orgDataSet;

    for(var k in temp){
      this.dataSet.push({input:this.formatStrings(temp[k].input),output:this.formatStrings(temp[k].output)})
      if(this.maxInputLength < temp[k].input.length) this.maxInputLength = temp[k].input.length;
      if(this.maxOutputLength < temp[k].output.length) this.maxOutputLength = temp[k].output.length;
      if(k == temp.length-1)this.fluffData();
    }
  },
  fluffData:function(){
    let temp = this.dataSet;
    for(var k in temp){
      if(this.maxInputLength > temp[k].input.length){
        for(var i = 0, j = this.maxInputLength - temp[k].input.length; i<j ; ++i){
          temp[k].input.push(0.00000);
        }
      }
      if(this.maxOutputLength > temp[k].output.length){
        for(var i = 0, p = this.maxOutputLength - temp[k].output.length; i<p ; ++i){
          temp[k].output.push(0.00000);
        }
      }
      if(k == temp.length-1)this.teach();
    }
  },
  fluffQuestion:function(questArr){
    if(this.maxInputLength > questArr.length){
        for(var i = 0, j = this.maxInputLength - questArr.length; i<j ; ++i){
          questArr.push(0.00000);
          if(questArr.length == this.maxInputLength) return questArr
        }
    }
    
  },
  convertAnswer:function(ansArr){
    let theString = '';
    for(var i = 0 ; i<ansArr.length;i++){
      let tempCharCode = ansArr[i]*255;
      if(31 < tempCharCode && tempCharCode < 128){
        theString = theString + String.fromCharCode(tempCharCode);
      }else{
        theString = theString + ' ';
      }
      if(i == ansArr.length-1)console.log(theString);
    }
    return theString
  },
  maxlengths: function (){
    return [this.maxInputLength,this.maxOutputLength];
  },

  setMaxlengths: function(inlen, outlen){
    if(inlen && outlen){
      this.maxOutputLength = outlen;
      this.maxInputLength = inlen;
    } 
    if(!inlen && outlen) this.maxOutputLength = outlen;
    if(!outlen && inlen) this.maxInputLength = inlen;
  },
  /*
  *returns{array of ascii numbers / 255 }
  */
  teach: function(opts){
    if(!this.nueral){
      this.nueral = new brain.NeuralNetwork({
        hiddenLayers:[178,105,240,100,188,122]
      });
    }
    console.log("time to learn!")
    
    let trainInfo = this.nueral.train(this.dataSet,{
      errorThresh: 0.00005,  // error threshold to reach
      iterations: 50000,   // maximum training iterations
      log: true,           // console.log() progress periodically
      logPeriod: 50,       // number of iterations between logging
      learningRate: 0.3    // learning rate

    });
    
    
    let teststring = this.fluffQuestion(this.formatStrings("what is your name?")); 
    let result = this.nueral.run(teststring)
    console.log(trainInfo);
    this.convertAnswer(result);
    this.trainedNet = this.nueral.toJSON();
  },
  answer: function(string){
    let tester = this.fluffQuestion(this.formatStrings(string));
    let output = this.nueral.run(tester);
    this.convertAnswer(output);
  },
  formatStrings: function (stringData){
    let charArr=[];
    for(let i = 0 ; i < stringData.length;i++){
      charArr.push(stringData.charCodeAt(i)/255);
    }
    return charArr;
  }
}

module.exports = BrainTeacher;