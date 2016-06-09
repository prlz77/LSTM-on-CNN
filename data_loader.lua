require 'dp'
require 'hdf5'
local class = require 'class'
SequentialDB = class('SequentialDB')


function SequentialDB:__init(dataPath, batchSize, rho)
  self.db = hdf5.open(dataPath, 'r')
  self.data = self.db:read('data')
  self.dim = self.data:dataspaceSize()
  self.seqNums = self.db:read('seq')
  self.labels = self.db:read('labels')
  self.bs = batchSize
  self.rho = rho
  self.seqStart = {}
  print("Counting sequence start frames...")
  for i = 1, self.dim[1] do
    local seq = self.seqNums:partial({i,i})
    if self.seqStart[seq] == nil then
      self.seqStart[seq] = i
    end
  end
  print("Generating random batch indices...")
  self.batchIndexs = torch.IntTensor(self.bs)
  for i = 1,self.bs do
    self.batchIndexs = torch.random(1,dim[1] - rho)
  end
end

function SequentialDB:getBatch()
  for i = 1,self.bs do
    local seqNum = self.seqNums:partial({self.batchIndexs[i],self.batchIndexs[i]})

  end

end


function load_db(train_path, val_path)
  local trainDB = hdf5.open(train_path,'r')
  local valDB = hdf5.open(val_path,'r')
  return trainDB, valDB
end


-- prepare batch offsets
offsets = {}
for i=1,opt.batchSize do
   table.insert(offsets, math.ceil(math.random()*train_inputs:size(1)))
end

function get_data(train_path, val_path, batchSize, seq_length)
  print("Loading dataset")
  print("Batch size: "..batchSize)
  print("Seq length:"..seq_length)
  local train_set
  train_set = hdf5.open(train_path, 'r')
  local seq_numbers = {}
  sequence_numbers_train = train_set:read('seq_number'):all()
  local train_outputs
  train_outputs = train_set:read('outputs'):all():squeeze()
  print("Data shape")
  print(train_outputs:size())
  local train_labels
  train_labels = train_set:read('labels'):all():squeeze():ge(1):add(1)
  train_set:close()
  print('New train outputs shape:')
  print(train_outputs:size())
  local val_set
  val_set = hdf5.open(val_path, 'r')
  sequence_numbers_val = val_set:read('seq_number'):all()
  local val_outputs
  val_outputs = val_set:read('outputs'):all():squeeze()
  local val_labels
  val_labels = val_set:read('labels'):all():ge(1):add(1)
  batch_outputs = {}
  batch_labels = {}
  current_seq = 0
  seq_start = 1
  seq_offset = 0
  inputs = {}
  targets = {}
  while seq_start + seq_offset + seq_length - 1 <= val_outputs:size(1) do
    if sequence_numbers_val[seq_start + seq_offset + seq_length - 1] ~= current_seq then
      current_seq = sequence_numbers_val[seq_start + seq_offset + seq_length - 1]
      seq_start = seq_start + seq_offset + seq_length - 1
      seq_offset = 0
    end
    table.insert(batch_outputs, val_outputs:sub(seq_start + seq_offset, seq_start + seq_offset + seq_length - 1):view(1, seq_length, -1):double())
    table.insert(batch_labels, val_labels:sub(seq_start + seq_offset, seq_start + seq_offset + seq_length - 1):view(1, seq_length, -1):double())
    seq_offset = seq_offset + 1
  end
  collectgarbage()
  batch_outputs = torch.cat(batch_outputs, 1):float():transpose(1,2):contiguous():cuda():split(1,1)
  batch_labels = torch.cat(batch_labels, 1):float():transpose(1,2):contiguous():cuda():split(1,1)
  collectgarbage()
  for i,v in ipairs(batch_outputs) do
    batch_outputs[i] = batch_outputs[i]:view(batch_outputs[i]:size(2), batch_outputs[i]:size(3))
    batch_labels[i] = batch_labels[i]:view(batch_labels[i]:size(2), batch_labels[i]:size(3))
  end
  val_set:close()
  collectgarbage()
  return train_outputs, train_labels, sequence_numbers_train, batch_outputs, batch_labels, sequence_numbers_val
end

function get_shaped_data(train_path, val_path, batchSize, seq_length)
--function get_features(train_path, val_path, seq_length)
  print("Loading dataset")
  print("Batch size: "..batchSize)
  print("Seq length:"..seq_length)
  local train_set
  train_set = hdf5.open(train_path, 'r')
  local train_outputs
  train_outputs = train_set:read('outputs'):all()
  print("Data shape")
  print(train_outputs:size())
  local remainder
  remainder = train_outputs:size(1) - (train_outputs:size(1) % (batchSize*seq_length))
  --print("Remainder: "..(remainder/(batchSize*seq_length)))

  train_outputs = train_outputs:narrow(1, 1, remainder)
  local train_labels
  train_labels = train_set:read('labels'):all()
  train_labels = train_labels:narrow(1, 1, remainder)
  train_set:close()
  --train_outputs = train_outputs:view(-1, batchSize, seq_length, train_outputs:size(3))
  train_outputs = train_outputs:view(-1, seq_length, train_outputs:size(3))
  local shuffle
  shuffle = torch.LongTensor(train_outputs:size(1))
  shuffle[{}] = torch.randperm(train_outputs:size(1))
  train_outputs = train_outputs:index(1,shuffle)
  train_outputs = train_outputs:view(-1, batchSize, seq_length, train_outputs:size(3))

  train_labels = train_labels:view(-1, seq_length, 1)
  train_labels = train_labels:index(1, shuffle)
  train_labels = train_labels:view(-1, batchSize, seq_length, 1)
  print('New train outputs shape:')
  print(train_outputs:size())
  local val_set
  val_set = hdf5.open(val_path, 'r')
  local val_outputs
  val_outputs = val_set:read('outputs'):all()
  remainder = val_outputs:size(1) - (val_outputs:size(1) % (batchSize*seq_length))
  val_outputs = val_outputs:narrow(1, 1, remainder)
  local val_labels
  val_labels = val_set:read('labels'):all()
  val_labels = val_labels:narrow(1, 1, remainder)
  val_set:close()
  val_outputs = val_outputs:view(-1, batchSize, seq_length, val_outputs:size(3))
  val_labels = val_labels:view(-1, batchSize, seq_length, 1)
  print('New val outputs shape:')
  print(val_outputs:size())
  train_outputs = train_outputs:transpose(2,3):contiguous()
  train_labels = train_labels:transpose(2,3):contiguous():ge(1):add(1)
  val_outputs = val_outputs:transpose(2,3):contiguous()
  val_labels = val_labels:transpose(2,3):contiguous():ge(1):add(1)
  collectgarbage()
  return train_outputs, train_labels, val_outputs, val_labels
end

function test_data_loader()
  local train_outputs, train_labels, val_outputs, val_labels
  train_outputs, train_labels, val_outputs, val_labels = get_data('../vgg_face_balanced_box_euclidean/fc6.h5', '../vgg_face_balanced_box_euclidean/fc6_val.h5', 2, 3)
end
--test_data_loader()
