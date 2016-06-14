-- Simple test for the data_loader.
-- Author: Pau Rodríguez López (@prlz77)
-- Mail: pau.rodri1 at gmail.com
-- Institution: ISELAB in CVC-UAB
-- Date: 14/06/2016

require 'data_loader'
require 'hdf5'
require 'xlua'
require 'math'
MIN_SEQ = 100
MAX_BS = 100
MAX_ITER = 10
print('Testing...')
for it = 1, MAX_ITER do
    local rho = torch.random(1, 100)
    local bs = torch.random(1,math.min(MAX_BS, math.floor(200/rho)))
    print('Batch Size: '..bs)
    print('Rho: '..rho)
    local dataLoader = SequentialDB('./tests/toydata.h5', bs, rho)
    local i_min = dataLoader.batchIndexs:clone() - 1
    local i_max = i_min:clone() + dataLoader.rho - 1

    for i = 1,dataLoader.bs do
        if i_max[i] > 99 then
            i_min[i] = 500 + math.floor(i_max[i] % 100)
            i_max[i] = i_min[i] + dataLoader.rho - 1
        end
    end
    for it2 = 1,200 do
        data, labels = dataLoader:getBatch()
        for i = 1, dataLoader.bs do
            local seq = torch.Tensor({i_min[i]})
            if i_min[i] ~= i_max[i] then
                seq = torch.linspace(i_min[i], i_max[i], dataLoader.rho)
            end
            assert(data[{i}]:eq(seq):all())
            local y = seq+1000
            assert((y - labels[{i}]):abs():sum() < 0.00001)
        end
        i_min:add(bs)
        i_max:add(bs)
        for i = 1,dataLoader.bs do
            repeat
                local i_min2  = i_min:clone()
                local i_max2 = i_max:clone()
                if i_max[i] > 99 and i_min[i] < 500 then
                    i_min[i] = 500 + math.floor(i_max[i] % 100)
                    i_max[i] = i_min[i] + dataLoader.rho - 1
                elseif i_max[i] > 599 and i_min[i] >= 500 then
                    i_min[i] = math.floor(i_max[i] % 100)
                    i_max[i] = i_min[i] + dataLoader.rho - 1
                end
            until i_min2:eq(i_min):all() and i_max2:eq(i_max):all()
        end
        xlua.progress(it2,200)
    end
    print('OK.')
end

