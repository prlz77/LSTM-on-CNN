-- LSTM training framework for hdf5 data
-- Author: Pau Rodríguez López (@prlz77)
-- Mail: pau.rodri1 at gmail.com
-- Institution: ISELAB in CVC-UAB
-- Date: 14/06/2016
-- Description: Performs regression with a LSTM neural network on arbitrary sequential hdf5 data in the form n-to-one. 
-- It also has plotting features and allows to save outputs for further processing.

require 'rnn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'optim'
require 'data_loader'
require 'paths'
require 'gnuplot'
require 'math'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a dataset using LSTM')
cmd:text('Example:')
cmd:text("LSTM.lua --rho 10")
cmd:text('Options:')
cmd:option('--trainPath', '', 'train.h5 path')
cmd:option('--valPath', '', 'validation.h5 path')
cmd:option('--learningRate', 0.01, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--batchSize', 32, 'number of examples per batch')

-- model i/0
cmd:option('--load', '', 'Load LSTM pre-trained weights')
cmd:option('--saveOutputs', '', '.h5 file path to save outputs')
cmd:option('--saveBestOutputs', '', '.h5 file path to save best outputs')

--cmd:option('--cuda', false, 'use CUDA')
--cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
--cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 200, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--depth', 1, 'number of hidden layers')
--cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- loss
cmd:option('--task', 'regress', 'main LSTM task [regress | classify]')
cmd:option('--nlabels', 0, 'number of output neurons (max(labels) by default)')

-- other
cmd:option('--printEvery', 0, 'print loss every n iters')
cmd:option('--testEvery', 1, 'print test accuracy every n epochs')
cmd:option('--logPath', '', 'log here')
cmd:option('--savePath', './snapshots', 'save snapshots here')
cmd:option('--saveEvery', 0, 'number of epochs to save model snapshot')
cmd:option('--plotRegression', 0, 'number of epochs to plot regression approximation')
cmd:option('--testOnly', false, 'Test only flag')
cmd:option('--auc', false, 'Save auc flag')
cmd:text()

opt = cmd:parse(arg or {})

-- snapshots folder
if opt.saveEvery ~= 0 then
  opt.savePath=paths.concat(opt.savePath, os.date("%d_%m_%y-%T"))
  paths.mkdir(opt.savePath)
end

--log path settings
if opt.logPath == '' then
  paths.mkdir('./logs')
  opt.logPath = paths.concat('./logs', os.date("%d_%m_%y-%T")..'.log')
else
  paths.mkdir(paths.dirname(opt.logPath))
end

-- initialize dataset
local hdf5_fields = {data='outputs', labels='labels', seq='seq_number'}
local trainDB = SequentialDB(opt.trainPath, opt.batchSize, opt.rho, false, hdf5_fields)
local valDB = SequentialDB(opt.valPath, opt.batchSize, opt.rho, false, hdf5_fields) --bs=1 to loop only once through all the data.
local trainIters = math.floor(trainDB.N / trainDB.bs)
local valIters = math.floor(valDB.N / valDB.bs)
--valDB.batchIndexs = torch.linspace(1,opt.batchSize, opt.batchSize)
local dataDim = trainDB.dim[2]*trainDB.dim[3]*trainDB.dim[4] -- get flat data dimensions
-- start logger
logger = optim.Logger(opt.logPath)
if opt.task == 'regress' then
    if opt.auc then
	  logger:setNames{'epoch', 'train error', 'test error', 'auc'}
    else
      logger:setNames{'epoch', 'train error', 'test error'}
    end
else
	logger:setNames{'epoch', 'train error', 'test error', 'accuracy'}
end

if opt.load == '' then
  -- turn on recurrent batchnorm
  nn.FastLSTM.bn = true
  -- build LSTM RNN
  rnn = nn.Sequential()
  rnn:add(nn.SplitTable(1,2)) -- (bs, rho, dim)
  rnn = rnn:add(nn.Sequencer(nn.FastLSTM(dataDim, opt.hiddenSize)))
  if opt.dropoutProb > 0 then
    rnn = rnn:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
  end

  for d = 1,(opt.depth - 1) do
    rnn = rnn:add(nn.Sequencer(nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)))
    if opt.dropoutProb > 0 then
      rnn = rnn:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
    end
  end
  if opt.task == 'regress' then
  	rnn = rnn:add(nn.Sequencer(nn.Linear(opt.hiddenSize, trainDB.ldim[2])))
  else
  	local nlabels
  	if opt.nlabels > 0 then
  		nlabels = opt.nlabels
  	else
  		nlabels = trainDB.maxLabel
  	end
  	rnn = rnn:add(nn.Sequencer(nn.Linear(opt.hiddenSize, nlabels)))
  end
  rnn:add(nn.SelectTable(-1))

  -- CPU -> GPU
  rnn:cuda()

  -- random init weights
  for k,param in ipairs(rnn:parameters()) do
    param:uniform(-opt.uniform, opt.uniform)
  end

else --pre-trained model
  rnn = torch.load(opt.model)
end

-- show the network
print(rnn)

-- build criterion
local criterion
if opt.task == 'regress' then
    criterion = nn.MSECriterion():cuda()
else
    criterion = nn.CrossEntropyCriterion():cuda()
end
print(criterion)

-- optimizer state
local optimState = {learningRate = opt.learningRate}

parameters, gradParameters = rnn:getParameters()

-- save only the necessary values
lightModel = rnn:clone('weight','bias','running_mean','running_std')

--set current epoch
local epoch = 1

function train()
  rnn:training()

  local feval = function(x)
    if x ~= parameters then parameters:copy(x) end
    gradParameters:zero()
    inputs, targets = trainDB:getBatch()
    inputs = inputs:resize(opt.batchSize,opt.rho,dataDim):cuda()
    targets = targets[{{},-1,{}}]:resize(opt.batchSize, valDB.ldim[2]):cuda()
    outputs = rnn:forward(inputs)
    local f = criterion:forward(outputs, targets)
    local df_do = criterion:backward(outputs, targets)
    rnn:backward(inputs, df_do)
    --clip gradients
    rnn:gradParamClip(5)
    return f,gradParameters
  end
  -- keep avg loss
  local loss = 0
  for iter = 1, trainIters do
    parameters, f = optim.adam(feval, parameters, optimState)
    xlua.progress(iter, trainIters)
    if iter % opt.printEvery == 0 then
      print('Iter: '..iter..', loss: '..loss )
    end
    loss = loss + f[1]
  end
  return loss / trainIters
end

local auc = function(outputs, targets)
 -- sort the scores:
         if not outputs:nElement() == 0 then return 0.5 end
         local scores, sortind = torch.sort(outputs, 1, true)

         -- construct the ROC curve:
         local tpr = torch.DoubleTensor(outputs:nElement() + 1):zero()
         local fpr = torch.DoubleTensor(outputs:nElement() + 1):zero()
         for n = 2,scores:nElement() + 1 do
            if targets[sortind[n - 1]] == 1 then
               tpr[n], fpr[n] = tpr[n - 1] + 1, fpr[n - 1]
            else
               tpr[n], fpr[n] = tpr[n - 1], fpr[n - 1] + 1
            end
         end
         tpr:div(targets:sum())
         fpr:div(torch.mul(targets, -1):add(1):sum())

         -- compute AUC:
         local auc = torch.cmul(
            tpr:narrow(1, 1, tpr:nElement() - 1),
            fpr:narrow(1, 2, fpr:nElement() - 1) -
            fpr:narrow(1, 1, fpr:nElement() - 1)):sum()

         -- return AUC and ROC curve:
         return auc, tpr, fpr
end

aucScore = 0
local bestAuc = -1

function test()
  rnn:evaluate()
  -- keep avg loss
  local loss = 0
  local outputHist = {}
  local targetHist = {}
  local saveHist = (opt.plotRegression ~= 0 or opt.auc or opt.saveOutputs ~= '' or opt.saveBestOutputs ~= '' )
  accuracy = 0
  --local inputHist = {} uncomment if 1D
  for iter = 1, valIters do
    inputs, targets = valDB:getBatch()
    inputs = inputs:resize(valDB.bs,opt.rho,dataDim):cuda()
    targets = targets[{{},-1,{}}]:resize(valDB.bs, valDB.ldim[2]):cuda()
    local outputs = rnn:forward(inputs)
    if opt.task == 'classify' then
    	max, ind = torch.max(outputs, 2)
    	accuracy = accuracy + ind:eq(targets):sum() / outputs:size(1)
    end
    if saveHist then
      --inputHist[iter] = inputs[{{},-1,{}}]:float():view(-1) uncomment if 1D
      outputHist[iter] = outputs:float():view(-1)
      targetHist[iter] = targets:float():view(-1)
    end
    local f = criterion:forward(outputs, targets)    
    xlua.progress(iter, valIters)
    loss = loss + f
  end
  if opt.task == 'classify' then
  	print('Accuracy ' .. accuracy / valIters)
  end
  if saveHist then
    --inputHist = nn.JoinTable(1,1):forward(inputHist) uncomment if 1D
    -- edge efects if rho > 1 because we need rho frames to predict the last one
    outputHist_join = nn.JoinTable(1,1):forward(outputHist)
    targetHist_join = nn.JoinTable(1,1):forward(targetHist)
    if opt.auc then
      aucScore, tpr, fpr = auc(outputHist_join, targetHist_join:gt(0))
      print('Auc:' .. aucScore)
    end
    if (epoch % opt.plotRegression) == 0 then
      gnuplot.plot({'outputs', outputHist_join, '-'},{'targets', targetHist_join, '-'})
    end
    --gnuplot.plot({'inputs', inputHist, '.'},{'outputs', outputHist, '-'},{'targets', targetHist, '-'}) --uncomment if 1D
        
    if opt.saveOutputs ~= '' then
      local output = hdf5.open(opt.saveOutputs, 'w')
      output:write('outputs', outputHist_join)
      output:write('labels', targetHist_join)
      output:close()
    end

    if opt.saveBestOutputs ~= '' and bestAuc < aucScore then
      bestAuc = aucScore
      local output = hdf5.open(opt.saveBestOutputs, 'w')
      output:write('outputs', outputHist_join)
      output:write('labels', targetHist_join)
      output:close()
    end
  end
      
  return loss / valIters, outputs
end

while epoch < opt.maxEpoch do
  local trainLoss = nil
  if not opt.testOnly then
    print('epoch '..epoch..':')
    print('Train:')
    trainLoss = train()
    print('Avg train loss: '..trainLoss)
  end
  local testLoss = nil
  if (epoch % opt.testEvery) == 0 then
    print('Test:')
    testLoss = test()
    print('Avg test loss: '..testLoss)
  end
  if opt.task == 'regress' then
    if opt.auc then
      logger:add({epoch, trainLoss, testLoss, aucScore})
    else
      logger:add({epoch, trainLoss, testLoss})
    end
  else
    logger:add({epoch, trainLoss, testLoss, accuracy})
  end
  if (epoch % opt.saveEvery) == 0 then
    torch.save(opt.savePath..'/model_'..epoch..'.t7',lightModel)
  end
  epoch = epoch + 1
  collectgarbage()
end
