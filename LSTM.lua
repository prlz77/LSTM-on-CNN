require 'rnn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'data'
require 'optim'
require 'data_loader'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a dataset using LSTM')
cmd:text('Example:')
cmd:text("LSTM.lua --rho 10")
cmd:text('Options:')
cmd:option('--trainPath', '', 'train.h5 path')
cmd:option('--valPath', '', 'validation.h5 path')
cmd:option('--inputSize',1, 'Input size')
cmd:option('--learningRate', 0.05, 'learning rate at t=0')
cmd:option('--momentum', 0.9, 'momentum')
cmd:option('--maxOutNorm', -1, 'max l2-norm of each layer\'s output neuron weights')
cmd:option('--cutoffNorm', -1, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('--batchSize', 32, 'number of examples per batch')
--cmd:option('--cuda', false, 'use CUDA')
--cmd:option('--useDevice', 1, 'sets the device (GPU) to use')
cmd:option('--maxEpoch', 1000, 'maximum number of epochs to run')
cmd:option('--maxTries', 50, 'maximum number of epochs to try to find a better local minima for early-stopping')
--cmd:option('--progress', false, 'print progress bar')
--cmd:option('--silent', false, 'don\'t print anything to stdout')
cmd:option('--uniform', 0.1, 'initialize parameters using uniform distribution between -uniform and uniform. -1 means default initialization')

-- recurrent layer
cmd:option('--rho', 5, 'back-propagate through time (BPTT) for rho time-steps')
cmd:option('--hiddenSize', 200, 'number of hidden units used at output of each recurrent layer. When more than one is specified, RNN/LSTMs/GRUs are stacked')
cmd:option('--depth', 1, 'number of hidden layers')
--cmd:option('--zeroFirst', false, 'first step will forward zero through recurrence (i.e. add bias of recurrence). As opposed to learning bias specifically for first step.')
--cmd:option('--dropout', false, 'apply dropout after each recurrent layer')
cmd:option('--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')

-- other
cmd:option('--printEvery', 10, 'print loss every n iters')
cmd:option('--testEvery', 10, 'print test accuracy every n iters')
cmd:option('--logPath', './log.txt', 'log here')

cmd:text()
opt = cmd:parse(arg or {})


-- init log
log = io.open(opt.logPath, 'w')


-- turn on recurrent batchnorm
nn.FastLSTM.bn = true
-- build LSTM RNN
local rnn = nn.Sequential()
rnn:add(nn.SplitTable(1,4))
rnn = rnn:add(nn.Sequencer(nn.FastLSTM(opt.inputSize, opt.hiddenSize)))
if opt.dropoutProb > 0 then
  rnn = rnn:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
end

for d = 1,(opt.depth - 1) do
  rnn = rnn:add(nn.Sequencer(nn.FastLSTM(opt.hiddenSize, opt.hiddenSize)))
  if opt.dropoutProb > 0 then
    rnn = rnn:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
  end
end
rnn = rnn:add(nn.Sequencer(nn.Linear(opt.hiddenSize, opt.nClasses)))
rnn:add(nn.SelectTable(-1))

-- CPU -> GPU
rnn:cuda()

-- random init weights
for k,param in ipairs(rnn:parameters()) do
  param:uniform(-opt.uniform, opt.uniform)
end

-- show the network
print(rnn)

-- build criterion
local criterion = nn.MSECriterion():cuda()

-- optimizer state
local optimState = {learningRate = opt.learningRate}

-- initialize dataset
local train = sequentialDB(opt.trainPath, opt.batchSize, opt.rho)
local val = sequentialDB(opt.valPath, opt.batchSize, opt.rho)

parameters, gradParameters = rnn:getParameters()

function train()
  rnn:training()

  local feval = function(x)
    if x ~= parameters then parameters:copy(x) end
    gradParameters:zero()
    inputs, targets = train:getBatch()
    inputs:cuda()
    targets:cuda()
    local outputs = model:forward(inputs)
    local f = criterion:forward(outputs, targets[{-1}]:view(opt.batchSize, -1))
    local df_do = criterion:backward(outputs, targets[{-1}]:view(opt.batchSize, -1))
    model:backward(inputs, df_do)
    --clip gradients
    rnn:gradParamClip(5)
    return f,gradParameters
  end
  -- keep avg loss
  local loss = 0
  for iter = 1, train.dim[1] do
    parameters, f = optim.adam(feval, parameters, optimState)
    xlua.progress(iter, train.dim[1])
    loss = loss + f
  end
  return loss
end

local epoch = 1
while epoch < opt.maxEpoch do
  print('epoch '..epoch..':')
  print('Train:')
  loss = train()
  print('Avg loss: '..loss)
  print('Test:')
  loss = test()
  print('Avg loss: '..loss)
  print('\n')
end