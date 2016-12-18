//index.js

var assert = require('assert');
var brain = require('./brain.js/dist/brain');
var net = new brain.NeuralNetwork();
var xor = [
  { input: {them:0.5, the:1, there:0.5, this:0.25 , that:0.25 , their:0.5, tit:0.05, time:0.05,television:0.05, alpha:0}, output: {t:1,the:1, th:1}},
  { input: {them:0.05, the:0.45, there:0.05, this:0.15 , that:0.05 , their:0.15, tit:1 , time:0.50,television:0.05, alpha:0}, output: {t:1,tit:1, ti:1}},
  { input: {them:0.05, the:0.15, there:0.5, this:0.5 , that:0.15 , their:0.5, tit:0.5 , time:1,television:0.05, alpha:0}, output: {t:1,time:1, ti:1}},
  { input: {them:0.05, the:0.05, there:0.05, this:0.05 , that:0.05 , their:0.05, tit:0.05 , time:0.05,television:1, alpha:0}, output: {t:1,television:1, te:1}}
];

net.train(xor,{
  errorThresh: 0.005,  // error threshold to reach
  iterations: 20000,   // maximum training iterations
  log: true,           // console.log() progress periodically
  logPeriod: 100,       // number of iterations between logging
  learningRate: 0.3    // learning rate
});
var output = net.run({the:1, them:1, alpha:1, bandit:0.6, townhouse:0.5}); 
console.log(output); 