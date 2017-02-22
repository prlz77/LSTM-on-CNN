require 'hdf5'
require 'nn'
require 'xlua'
require 'math'
require 'masked_data_loader'
MIN_SEQ = 100
MAX_BS = 100
MAX_ITER = 10
print('Testing...')
f = hdf5.open('./tests/toydata.h5','r')
data = f:read('outputs'):all()
seq = f:read('seq'):all()
label = f:read('labels'):all()
batch_data = {}
batch_labels = {}
current_seq = 2
data_buffer = {}
label_buffer = {}
for i = 1,200 do
    if seq[i] ~= current_seq or i == 200 then
        if i == 200 then
            table.insert(data_buffer, data[{i,1,1,1}])
            table.insert(label_buffer, label[{i,1}])
        end
        while #data_buffer < 10 do
            table.insert(data_buffer, 1, 0)
            table.insert(label_buffer, 1, 0)
        end
        current_seq = seq[i]
        table.insert(batch_data, data_buffer)
        table.insert(batch_labels, label_buffer)
        data_buffer = {}
        label_buffer = {}
    end
    if i ~= 200 then
        table.insert(data_buffer, data[{i,1,1,1}])
        table.insert(label_buffer, label[{i,1}])
    end
end

batch_data = torch.Tensor(batch_data)
batch_labels = torch.Tensor(batch_labels)
batch_data = batch_data:view(-1,10,1,1,1)
batch_labels = batch_labels:view(-1, 10, 1)

for it = 1, MAX_ITER do
    local bs = torch.random(1,math.min(MAX_BS, batch_data:size(1)))
    print('Batch Size: '..bs)
    local dataLoader = SequentialDB('./tests/toydata.h5', bs, false, {data='outputs', labels='labels', seq='seq'})
    assert(dataLoader.rho == 10)
    local batchIndices = torch.linspace(1, bs, bs)
    _data, _labels = dataLoader:getBatch() 
    assert(_data:size(1) == bs)
    assert(_data:size(2) == 10)
    assert((_data:size(3)*_data:size(4)*_data:size(5)) == 1)
    dataLoader:reset()
    for it2 = 1, 200 do
      _data, _labels = dataLoader:getBatch()
      for it3 = 1,bs do
        b_index = batchIndices[it3]
        --print(_data[{it3}])
        --print(batch_data[{b_index}])
        assert(torch.eq(_data[{it3}], batch_data[{b_index}]):all()) 
        assert(torch.eq(_labels[{it3}], batch_labels[{b_index}]):all())
        batchIndices[it3] = 1 + ((b_index + bs - 1) % batch_data:size(1))
      end
      xlua.progress(it2, 200)
    end
    print('OK.')
end

