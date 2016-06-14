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

require 'dp'
require 'hdf5'
local class = require 'class'
SequentialDB = class('SequentialDB')


function SequentialDB:__init(dataPath, batchSize, rho, step)
  self.step = step or 1
  self.db = hdf5.open(dataPath, 'r')
  self.data = self.db:read('data')
  self.dim = self.data:dataspaceSize()
  self.seqNums = self.db:read('seq')
  self.labels = self.db:read('labels')
  self.ldim = self.labels:dataspaceSize()
  self.bs = batchSize
  self.rho = rho
  self.seqStart = {}
  self.dataTensor = torch.Tensor(batchSize, rho, self.dim[2], self.dim[3], self.dim[4])
  self.targetTensor = torch.Tensor(batchSize, rho, self.ldim[2])
  print("Counting sequence start frames...")
  for i = 1, self.dim[1] do
    local seq = self.seqNums:partial({i,i})[1]
    if self.seqStart[seq] == nil then
      self.seqStart[seq] = i
    end
  end
  print("Generating random batch indices...")
  self.batchIndexs = torch.IntTensor(self.bs)
  for i = 1,self.bs do
    self.batchIndexs[i] = torch.random(1,self.dim[1] - rho)
    if self.batchIndexs[i] + self.rho - 1 > self.dim[1] then
      self.batchIndexs[i] = 1
    end
    local seq = self.seqNums:partial({self.batchIndexs[i],self.batchIndexs[i]})[1]
    local seq2 = self.seqNums:partial({self.batchIndexs[i]+self.rho-1, self.batchIndexs[i]+self.rho-1})[1]
    if seq ~= seq2 then
      self.batchIndexs[i] = self.seqStart[seq2]
    end
  end
end

function SequentialDB:getBatch()
  for i = 1,self.bs do
    if self.batchIndexs[i] + self.rho - 1 > self.dim[1] then
      self.batchIndexs[i] = 1
    end
    local seq = self.seqNums:partial({self.batchIndexs[i],self.batchIndexs[i]})[1]
    local seq2 = self.seqNums:partial({self.batchIndexs[i]+self.rho-1, self.batchIndexs[i]+self.rho-1})[1]
    if seq ~= seq2 then
      self.batchIndexs[i] = self.seqStart[seq2]
    end
    self.dataTensor[{i}] = self.data:partial({self.batchIndexs[i],self.batchIndexs[i]+self.rho-1},{1,self.dim[2]},{1,self.dim[3]},{1,self.dim[4]})
    self.targetTensor[i] = self.labels:partial({self.batchIndexs[i],self.batchIndexs[i]+self.rho-1},{1,self.ldim[2]})
  end

  self.batchIndexs:add(self.step)
  return self.dataTensor, self.targetTensor
end