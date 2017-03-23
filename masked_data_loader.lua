-- Sequential data provider for hdf5 data.
-- Author: Pau Rodríguez López (@prlz77)
-- Mail: pau.rodri1 at gmail.com
-- Institution: ISELAB in CVC-UAB
-- Date: 14/06/2016
-- Description: Allows to read sequential data from an hdf5 file with batch interleaving.
--              The hdf5 file should contain the following datasets:
--                'data'   - a NxCxWxH float tensor with N samples with depth C and spatial dimensions W and H (W,H = 1 when no spatial data).
--                'labels' - a NxC float tensor with the labels of the N samples (C is usually 1 for regression). 
--                'seq'    - a Nx1 int tensor with the sequence number of each frame. For example, two concatenated videos of 3 and 4 frames each would
--                           produce: 1112222.

require 'hdf5'
require 'math'
local class = require 'class'
SequentialDB = class('SequentialDB')

function SequentialDB:__init(dataPath, batchSize, shuffle, hdf5_fields, offset)
  self.shuffle = false or shuffle
  hdf5_fields = hdf5_fields or {data='outputs', labels='labels', seq='seq_number'}
  self.db = hdf5.open(dataPath, 'r')
  self.data = self.db:read(hdf5_fields.data)
  self.dim = self.data:dataspaceSize()
  self.seqNums = self.db:read(hdf5_fields.seq)
  self.labels = self.db:read(hdf5_fields.labels)
  self.ldim = self.labels:dataspaceSize()
  self.bs = batchSize
  self.rho = 0 
  self.offset = offset or 0
  self.maxLabel = self.labels:all():max() + self.offset
  self.minLabel = self.labels:all():min() + self.offset
  print("Generating Sequences")
  self.sequences = {}
  local prevSeq = self.seqNums:partial({1,1})[1]
  local start = 1
  for i = 1, self.dim[1] do
    local seq = self.seqNums:partial({i,i})[1]
    if seq ~= prevSeq then
        table.insert(self.sequences, {start, i - 1})
        self.rho = math.max(self.rho, i - start) -- note that the end frame is i-1 not i.
        start = i
        prevSeq = seq
    end
  end
  table.insert(self.sequences, {start, self.dim[1]})
  self.rho = math.max(self.rho, 1 + self.dim[1] - start)
   
  self.N = #self.sequences
  self.bs = math.min(self.N, self.bs)

  self.sequences = torch.IntTensor(self.sequences)
  if self.shuffle then
      print("Shuffling...")
      local shuff = torch.randperm(self.N):long()
      self.sequences = self.sequences:index(1, shuff)
  end
  -- Output tensors
  self.dataTensor = torch.Tensor(batchSize, self.rho, self.dim[2], self.dim[3], self.dim[4])
  self.targetTensor = torch.Tensor(batchSize, self.rho, self.ldim[2])
  self.batchIndexs = torch.linspace(1, self.bs, self.bs)
end

-- Make sure the labels start by one
function SequentialDB:minLabelToOne()
    if self.offset == 0 then
        self.offset = 1 - self.minLabel
        self.maxLabel = self.maxLabel + self.offset
    else
        error("offset already set")
    end
    return self.offset == 0
end

function SequentialDB:getLabelWeights()
    if self.labelWeights == nil then
        self.labelWeights = torch.histc(self.labels:all(), self.maxLabel):float()
        self.labelWeights = self.labelWeights:div(self.labelWeights:sum())
    end
    return self.labelWeights
end

function SequentialDB:reset()
  self.batchIndexs = torch.linspace(1, self.bs, self.bs)
end

function SequentialDB:getBatch()
  self.dataTensor:zero()
  self.targetTensor:zero()
  for i = 1,self.bs do
    local seqInterval = {self.sequences[self.batchIndexs[i]][1], self.sequences[self.batchIndexs[i]][2]}
    local pad = self.rho - (seqInterval[2] - seqInterval[1] + 1) --zero padding
    self.dataTensor[{i, {pad + 1,self.rho}}] = self.data:partial(seqInterval,{1,self.dim[2]},{1,self.dim[3]},{1,self.dim[4]})
    self.targetTensor[{i, {pad + 1, self.rho}}] = self.labels:partial(seqInterval,{1,self.ldim[2]})
    self.batchIndexs[i] = 1 + ((self.batchIndexs[i] + self.bs - 1) % self.N)
  end
  if self.offset == 1 then
      self.targetTensor:add(1)
  end
  return self.dataTensor, self.targetTensor
end
